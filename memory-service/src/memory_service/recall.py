import re, logging
from datetime import datetime, timezone
from sqlite3 import Connection
from typing import Optional
import tiktoken
from memory_service.slots import is_singleton

logger = logging.getLogger(__name__)

# tiktoken encoder for token counting (o200k_base covers GPT-4o vocab)
try:
    _enc = tiktoken.get_encoding("o200k_base")
except Exception:
    _enc = None

LOW_SCORE_THRESHOLD = 0.15
PROFILE_SLOTS = {
    "identity.name", "identity.age", "identity.pronouns",
    "location.current", "location.previous", "location.hometown",
    "employment.current_company", "employment.current_role", "employment.previous_company",
    "relationship.partner", "preference.response_style", "preference.communication_style",
    "preference.diet",
}

def _count_tokens(text: str) -> int:
    if _enc is None:
        return len(text) // 4  # rough fallback: 4 chars ≈ 1 token
    return len(_enc.encode(text))


def _normalize_bm25(raw_scores: list[float]) -> list[float]:
    """Normalize BM25 scores to [0, 1] using min-max."""
    if not raw_scores:
        return []
    min_s = min(raw_scores)
    max_s = max(raw_scores)
    if max_s == min_s:
        return [1.0] * len(raw_scores)
    return [(s - min_s) / (max_s - min_s) for s in raw_scores]


def _recency_score(updated_at: str) -> float:
    """Score from 0-1 based on how recent the memory is. Decays over ~90 days."""
    try:
        dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        days_old = max(0, (now - dt).days)
        return max(0.0, 1.0 - days_old / 90.0)
    except Exception:
        return 0.5


def _build_fts_query(query: str) -> str:
    """Build a simple FTS5 query: drop stopwords, OR-join remaining terms."""
    STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                 "do", "does", "did", "has", "have", "had", "i", "you", "we",
                 "they", "it", "this", "that", "what", "who", "where", "when",
                 "how", "which", "and", "or", "but", "in", "on", "at", "to",
                 "for", "of", "with", "by", "from", "user", "the", "their"}
    words = re.findall(r"\b\w{2,}\b", query.lower())
    filtered = [w for w in words if w not in STOPWORDS]
    if not filtered:
        filtered = words  # if all stopwords, use all
    # FTS5 OR query — escape any special chars
    escaped = [re.sub(r'[^\w]', '', w) for w in filtered]
    escaped = [w for w in escaped if w]
    if not escaped:
        return "*"
    return " OR ".join(escaped)


def _get_profile_facts(conn: Connection, user_id: str) -> list[dict]:
    """Return active Tier-1 profile facts for user, ordered by slot."""
    rows = conn.execute(
        """SELECT m.id, m.slot, m.entity_key, m.value_text, m.confidence,
                  m.source_turn_id, m.updated_at, m.evidence, m.active,
                  -- get previous value if exists
                  (SELECT value_text FROM memories prev
                   WHERE prev.user_id=m.user_id AND prev.slot=m.slot
                     AND prev.active=0 AND prev.supersedes IS NULL
                     -- pick the most recent inactive one that was superseded
                     -- simplification: just get any inactive for same slot
                   LIMIT 1) as prior_value
           FROM memories m
           WHERE m.user_id=? AND m.active=1 AND m.slot IN ({})
           ORDER BY m.slot
        """.format(",".join("?" * len(PROFILE_SLOTS))),
        (user_id, *PROFILE_SLOTS)
    ).fetchall()
    return [dict(r) for r in rows]


def _get_relevant_memories(conn: Connection, user_id: str, fts_query: str, limit: int = 20) -> list[dict]:
    """BM25 search over FTS5 index, filtered to user's active memories."""
    try:
        rows = conn.execute(
            """SELECT m.id, m.slot, m.entity_key, m.value_text, m.confidence,
                      m.source_turn_id, m.updated_at, m.evidence, m.active,
                      bm25(memories_fts) AS bm25_raw
               FROM memories_fts
               JOIN memories m ON memories_fts.rowid = m.rowid
               WHERE memories_fts MATCH ?
                 AND m.user_id = ?
                 AND m.active = 1
               ORDER BY bm25(memories_fts)
               LIMIT ?""",
            (fts_query, user_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning("FTS search failed (query=%r): %s", fts_query, e)
        return []


def _get_recent_turns(conn: Connection, session_id: str, limit: int = 5) -> list[dict]:
    """Return recent turns for the session."""
    import json as _json
    rows = conn.execute(
        """SELECT id, messages_json, timestamp FROM turns
           WHERE session_id=? ORDER BY created_at DESC LIMIT ?""",
        (session_id, limit)
    ).fetchall()
    results = []
    for row in rows:
        try:
            msgs = _json.loads(row["messages_json"])
            user_msg = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            results.append({
                "id": row["id"],
                "timestamp": row["timestamp"],
                "snippet": user_msg[:200],
            })
        except Exception:
            pass
    return results


def _format_profile_section(profile_facts: list[dict]) -> tuple[str, list[dict]]:
    """Format Tier-1 profile facts into the '## Known facts' section."""
    if not profile_facts:
        return "", []

    # Build a deduplicated human-readable set grouped by category
    # (location.current and location.previous → "Lives in Berlin (previously NYC)")
    slot_values: dict[str, dict] = {}
    for f in profile_facts:
        slot_values[f["slot"]] = f

    lines = []
    citations = []
    processed = set()

    _SLOT_LABELS = {
        "identity.name": "Name",
        "identity.age": "Age",
        "identity.pronouns": "Pronouns",
        "location.current": "Lives in",
        "location.hometown": "Hometown",
        "employment.current_company": "Works at",
        "employment.current_role": "Role",
        "relationship.partner": "Partner",
        "preference.response_style": "Prefers",
        "preference.communication_style": "Communication style",
        "preference.diet": "Diet",
    }

    for slot, label in _SLOT_LABELS.items():
        if slot in slot_values and slot not in processed:
            f = slot_values[slot]
            val = f["value_text"]
            date_str = f["updated_at"][:10] if f.get("updated_at") else ""

            # Augment location.current with previous
            if slot == "location.current":
                prev = slot_values.get("location.previous")
                if prev:
                    val = f"{val} (previously {prev['value_text']})"
                    processed.add("location.previous")

            # Augment employment.current_company with previous
            if slot == "employment.current_company":
                role = slot_values.get("employment.current_role")
                prev_co = slot_values.get("employment.previous_company")
                if role:
                    val = f"{val} as {role['value_text']}"
                    processed.add("employment.current_role")
                if prev_co:
                    val = f"{val} (previously at {prev_co['value_text']})"
                    processed.add("employment.previous_company")

            line = f"- {label}: {val}."
            if date_str:
                line = f"- {label}: {val}. (updated {date_str})"
            lines.append(line)
            processed.add(slot)
            if f.get("source_turn_id"):
                citations.append({"turn_id": f["source_turn_id"], "score": 1.0, "snippet": f["evidence"] or val})

    # Any remaining profile facts not in the label map
    for slot, f in slot_values.items():
        if slot not in processed:
            lines.append(f"- {slot.replace('.', ' ').title()}: {f['value_text']}.")
            if f.get("source_turn_id"):
                citations.append({"turn_id": f["source_turn_id"], "score": 1.0, "snippet": f["evidence"] or f["value_text"]})

    return "## Known facts about this user\n" + "\n".join(lines), citations


def _format_relevant_section(memories: list[dict], profile_slot_ids: set[str]) -> tuple[str, list[dict]]:
    """Format query-relevant memories (excluding profile facts already shown)."""
    lines = []
    citations = []
    for m in memories:
        if m["id"] in profile_slot_ids:
            continue  # already in profile section
        ts = m.get("updated_at", "")[:10]
        val = m["value_text"]
        evidence = m.get("evidence", "")
        snippet = evidence if evidence else val
        lines.append(f"- [{ts}] {val}")
        if m.get("source_turn_id"):
            citations.append({
                "turn_id": m["source_turn_id"],
                "score": round(m.get("_final_score", 0.0), 4),
                "snippet": snippet[:200],
            })
    if not lines:
        return "", []
    return "## Relevant from past conversations\n" + "\n".join(lines), citations


def recall(
    conn: Connection,
    *,
    query: str,
    session_id: str,
    user_id: Optional[str],
    max_tokens: int = 1024,
) -> dict:
    """
    Main recall function. Returns {"context": str, "citations": list}.
    Never raises — returns empty on cold/unrelated queries.
    """
    if not user_id:
        user_id = f"anon:{session_id}"

    # Step 1: Always load profile facts (Tier-1, active)
    profile_facts = _get_profile_facts(conn, user_id)
    profile_ids = {f["id"] for f in profile_facts}

    # Step 2: BM25 search for query-relevant memories
    fts_query = _build_fts_query(query)
    relevant_rows = _get_relevant_memories(conn, user_id, fts_query, limit=20)

    # Normalize BM25 and compute final scores
    bm25_raws = [r["bm25_raw"] for r in relevant_rows]
    # BM25 from FTS5 is negative — negate so higher = better
    bm25_raws_pos = [-s for s in bm25_raws]
    bm25_norms = _normalize_bm25(bm25_raws_pos)

    for i, row in enumerate(relevant_rows):
        bm25_n = bm25_norms[i] if bm25_norms else 0.0
        recency = _recency_score(row.get("updated_at", ""))
        confidence = float(row.get("confidence", 1.0))
        active_boost = 1.0 if row.get("active") else 0.0
        score = 0.70 * bm25_n + 0.15 * recency + 0.10 * confidence + 0.05 * active_boost
        row["_final_score"] = score

    relevant_rows.sort(key=lambda r: r["_final_score"], reverse=True)

    # Noise resistance: if no profile facts AND top score < threshold → empty
    top_score = relevant_rows[0]["_final_score"] if relevant_rows else 0.0
    if not profile_facts and top_score < LOW_SCORE_THRESHOLD:
        return {"context": "", "citations": []}

    # Step 3: Recent session turns
    recent_turns = _get_recent_turns(conn, session_id, limit=5)

    # Step 4: Assemble context under token budget
    all_citations = []
    sections = []
    tokens_used = 0

    # Priority 1: profile facts
    profile_section, profile_citations = _format_profile_section(profile_facts)
    if profile_section:
        profile_tokens = _count_tokens(profile_section)
        if tokens_used + profile_tokens <= max_tokens * 2:  # soft cap: don't exceed 2×
            sections.append(profile_section)
            tokens_used += profile_tokens
            all_citations.extend(profile_citations)

    # Priority 2: query-relevant memories (greedy fill)
    RELEVANT_HEADER = "## Relevant from past conversations\n"
    relevant_header_tokens = _count_tokens(RELEVANT_HEADER)
    relevant_lines = []
    relevant_citations = []
    for m in relevant_rows:
        if m["id"] in profile_ids:
            continue
        ts = m.get("updated_at", "")[:10]
        val = m["value_text"]
        evidence = m.get("evidence", "")
        snippet = evidence if evidence else val
        line = f"- [{ts}] {val}"
        line_tokens = _count_tokens(line)
        # Reserve tokens for the section header on the first line
        header_overhead = relevant_header_tokens if not relevant_lines else 0
        if tokens_used + header_overhead + line_tokens > max_tokens * 2:
            break
        relevant_lines.append(line)
        tokens_used += header_overhead + line_tokens
        if m.get("source_turn_id"):
            relevant_citations.append({
                "turn_id": m["source_turn_id"],
                "score": round(m.get("_final_score", 0.0), 4),
                "snippet": snippet[:200],
            })

    if relevant_lines:
        sections.append("## Relevant from past conversations\n" + "\n".join(relevant_lines))
        all_citations.extend(relevant_citations)

    # Priority 3: recent session turns (if budget allows)
    recent_lines = []
    for turn in recent_turns[:2]:
        ts = turn["timestamp"][:10] if turn.get("timestamp") else ""
        snippet = turn["snippet"]
        line = f"- [{ts}] {snippet}"
        line_tokens = _count_tokens(line)
        if tokens_used + line_tokens > max_tokens * 2:
            break
        recent_lines.append(line)
        tokens_used += line_tokens

    if recent_lines:
        # Don't add a separate section — merge into relevant if relevant section exists
        if relevant_lines:
            sections[-1] += "\n" + "\n".join(recent_lines)
        else:
            sections.append("## Relevant from past conversations\n" + "\n".join(recent_lines))

    context = "\n\n".join(sections)
    return {"context": context, "citations": all_citations}


def search(
    conn: Connection,
    *,
    query: str,
    session_id: Optional[str],
    user_id: Optional[str],
    limit: int = 10,
) -> list[dict]:
    """
    Search memories. Returns structured results (not prose). Used by /search endpoint.
    """
    import json as _json
    if not user_id and not session_id:
        return []

    fts_query = _build_fts_query(query)

    # Build query dynamically based on available filters
    where_clauses = ["m.active = 1"]
    params = [fts_query]

    if user_id:
        where_clauses.append("m.user_id = ?")
        params.append(user_id)
    if session_id:
        where_clauses.append("m.source_session_id = ?")
        params.append(session_id)

    params.append(limit)

    try:
        rows = conn.execute(
            f"""SELECT m.id, m.slot, m.entity_key, m.value_text, m.confidence,
                       m.source_session_id, m.source_turn_id, m.updated_at,
                       m.evidence, m.attributes_json,
                       bm25(memories_fts) AS bm25_raw
                FROM memories_fts
                JOIN memories m ON memories_fts.rowid = m.rowid
                WHERE memories_fts MATCH ?
                  AND {' AND '.join(where_clauses)}
                ORDER BY bm25(memories_fts)
                LIMIT ?""",
            params
        ).fetchall()
    except Exception as e:
        logger.warning("Search FTS failed: %s", e)
        return []

    results = []
    for row in rows:
        try:
            attrs = _json.loads(row["attributes_json"] or "{}")
        except Exception:
            attrs = {}
        results.append({
            "content": row["value_text"],
            "score": abs(float(row["bm25_raw"])),
            "session_id": row["source_session_id"] or "",
            "timestamp": row["updated_at"],
            "metadata": attrs,
        })
    return results
