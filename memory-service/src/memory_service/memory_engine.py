import json, uuid, logging
from datetime import datetime, timezone
from sqlite3 import Connection
from memory_service.extractor.base import ExtractedMemory
from memory_service.slots import is_singleton, is_collection, get_previous_slot, SlotTier

logger = logging.getLogger(__name__)

LOW_CONFIDENCE_THRESHOLD = 0.45

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _new_id() -> str:
    return str(uuid.uuid4())

def get_user_context_summary(conn: Connection, user_id: str) -> str:
    """Return a compact string of active Tier-1 facts for LLM extraction context."""
    rows = conn.execute(
        "SELECT slot, value_text FROM memories WHERE user_id=? AND active=1",
        (user_id,)
    ).fetchall()
    if not rows:
        return ""
    return "\n".join(f"- {r['slot']}: {r['value_text']}" for r in rows)


def _deactivate(conn: Connection, memory_id: str, now: str) -> None:
    conn.execute(
        "UPDATE memories SET active=0, updated_at=? WHERE id=?",
        (now, memory_id)
    )


def _insert_memory(
    conn: Connection,
    *,
    user_id: str,
    source_session_id: str,
    source_turn_id: str,
    mem: ExtractedMemory,
    active: int,
    supersedes: str | None,
    now: str,
) -> str:
    mem_id = _new_id()
    conn.execute(
        """INSERT INTO memories
           (id, user_id, source_session_id, source_turn_id,
            type, slot, entity_key, value_text, attributes_json,
            confidence, evidence, active, supersedes, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            mem_id, user_id, source_session_id, source_turn_id,
            mem.type, mem.slot, mem.entity_key, mem.value_text,
            json.dumps(mem.attributes),
            mem.confidence, mem.evidence,
            active, supersedes,
            now, now,
        )
    )
    return mem_id


def _apply_singleton(
    conn: Connection,
    *,
    user_id: str,
    source_session_id: str,
    source_turn_id: str,
    mem: ExtractedMemory,
    now: str,
) -> str:
    """Apply a Tier-1 singleton memory with supersession."""
    existing = conn.execute(
        "SELECT id, value_text FROM memories WHERE user_id=? AND slot=? AND entity_key='' AND active=1",
        (user_id, mem.slot)
    ).fetchone()

    supersedes_id = None
    prior_value = None

    if existing:
        prior_value = existing["value_text"]
        if prior_value.strip().lower() == mem.value_text.strip().lower():
            logger.debug("Skipping duplicate singleton %s=%s for user %s", mem.slot, mem.value_text, user_id)
            return existing["id"]
        _deactivate(conn, existing["id"], now)
        supersedes_id = existing["id"]

    active = 1 if mem.confidence >= LOW_CONFIDENCE_THRESHOLD else 0
    new_id = _insert_memory(
        conn,
        user_id=user_id,
        source_session_id=source_session_id,
        source_turn_id=source_turn_id,
        mem=mem,
        active=active,
        supersedes=supersedes_id,
        now=now,
    )

    # Auto-write previous slot chain
    prev_slot = get_previous_slot(mem.slot)
    if prev_slot and prior_value and active == 1:
        # Write the prior value into the .previous slot (upsert)
        from memory_service.extractor.base import ExtractedMemory as EM
        prev_mem = EM(
            type="fact",
            slot=prev_slot,
            entity_key="",
            value_text=prior_value,
            confidence=0.95,
            evidence=f"auto-set from {mem.slot} update",
            mutation="upsert",
        )
        _apply_singleton(
            conn,
            user_id=user_id,
            source_session_id=source_session_id,
            source_turn_id=source_turn_id,
            mem=prev_mem,
            now=now,
        )

    return new_id


def _apply_collection(
    conn: Connection,
    *,
    user_id: str,
    source_session_id: str,
    source_turn_id: str,
    mem: ExtractedMemory,
    now: str,
) -> str:
    """Apply a Tier-2 collection memory (entity_key-aware)."""
    entity_key = mem.entity_key.strip().lower() if mem.entity_key else "unknown"
    existing = conn.execute(
        "SELECT id, value_text FROM memories WHERE user_id=? AND slot=? AND entity_key=? AND active=1",
        (user_id, mem.slot, entity_key)
    ).fetchone()

    supersedes_id = None
    if existing:
        if existing["value_text"].strip().lower() == mem.value_text.strip().lower():
            return existing["id"]
        _deactivate(conn, existing["id"], now)
        supersedes_id = existing["id"]

    # Use normalized entity_key
    mem_with_key = mem.model_copy(update={"entity_key": entity_key})
    active = 1 if mem.confidence >= LOW_CONFIDENCE_THRESHOLD else 0
    return _insert_memory(
        conn,
        user_id=user_id,
        source_session_id=source_session_id,
        source_turn_id=source_turn_id,
        mem=mem_with_key,
        active=active,
        supersedes=supersedes_id,
        now=now,
    )


def apply_memories(
    conn: Connection,
    *,
    user_id: str,
    source_session_id: str,
    source_turn_id: str,
    memories: list[ExtractedMemory],
) -> list[str]:
    """
    Apply extracted memory candidates to the database.
    Returns list of inserted/updated memory IDs.
    All operations run within the caller's connection (caller must commit).
    """
    if not user_id:
        user_id = f"anon:{source_session_id}"

    now = _now_iso()
    inserted_ids = []

    for mem in memories:
        try:
            if mem.mutation == "negate":
                # Deactivate all active memories matching this slot+entity_key
                rows = conn.execute(
                    "SELECT id FROM memories WHERE user_id=? AND slot=? AND entity_key=? AND active=1",
                    (user_id, mem.slot, mem.entity_key.lower())
                ).fetchall()
                for row in rows:
                    _deactivate(conn, row["id"], now)
                continue

            if is_singleton(mem.slot):
                mem_id = _apply_singleton(
                    conn, user_id=user_id,
                    source_session_id=source_session_id,
                    source_turn_id=source_turn_id,
                    mem=mem, now=now,
                )
                inserted_ids.append(mem_id)

            elif is_collection(mem.slot):
                mem_id = _apply_collection(
                    conn, user_id=user_id,
                    source_session_id=source_session_id,
                    source_turn_id=source_turn_id,
                    mem=mem, now=now,
                )
                inserted_ids.append(mem_id)

            else:
                # Tier 3 unstructured — always insert
                active = 1 if mem.confidence >= LOW_CONFIDENCE_THRESHOLD else 0
                mem_id = _insert_memory(
                    conn, user_id=user_id,
                    source_session_id=source_session_id,
                    source_turn_id=source_turn_id,
                    mem=mem, active=active, supersedes=None, now=now,
                )
                inserted_ids.append(mem_id)

        except Exception as e:
            logger.error("Failed to apply memory %s: %s", mem, e)
            # Don't crash the whole request; skip this candidate

    return inserted_ids
