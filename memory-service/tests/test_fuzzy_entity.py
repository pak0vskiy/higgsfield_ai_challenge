"""
Tests for fuzzy entity-key matching in _apply_collection.

Verifies that:
  - "mylo" and "mylo the cat" are merged (one active entity after both writes)
  - "mylo" and "biscuit" remain distinct (different entities)
  - Short keys (<3 chars) are not fuzzy-matched
  - The threshold (ENTITY_FUZZY_THRESHOLD) is respected
"""
import pytest
from memory_service.extractor.base import ExtractedMemory
from memory_service.memory_engine import apply_memories, _fuzzy_match_entity_key


def _pet(name: str, value: str, conf: float = 0.85) -> ExtractedMemory:
    return ExtractedMemory(
        type="fact",
        slot="pet",
        entity_key=name,
        value_text=value,
        confidence=conf,
        evidence=value,
        mutation="upsert",
    )


# ── Unit tests for _fuzzy_match_entity_key ─────────────────────────────────────

def test_fuzzy_match_same_name():
    """Identical keys always match."""
    result = _fuzzy_match_entity_key(["mylo"], "mylo")
    assert result == "mylo"


def test_fuzzy_match_variant_key():
    """'mylo the cat' should match existing key 'mylo'.
    token_set_ratio("mylo", "mylo the cat") = 100 since 'mylo' is a token subset.
    """
    result = _fuzzy_match_entity_key(["mylo"], "mylo the cat")
    assert result == "mylo", f"Expected 'mylo' match, got {result!r}"


def test_fuzzy_no_match_different_names():
    """'biscuit' should NOT match 'mylo'."""
    result = _fuzzy_match_entity_key(["mylo"], "biscuit")
    assert result is None


def test_fuzzy_no_match_short_key():
    """Keys shorter than 3 chars are excluded from fuzzy matching."""
    result = _fuzzy_match_entity_key(["ab"], "abc")
    assert result is None, "Short key 'ab' should not be fuzzy-matched"


def test_fuzzy_multiple_candidates_returns_best():
    """When multiple candidates exist, best-scoring one is returned.
    token_set_ratio('mylo luna', 'mylo') = 100 since 'mylo' is a subset,
    but 'mylo' exact-token-set match should beat 'biscuit' which has no overlap.
    """
    candidates = ["mylo", "biscuit", "luna"]
    result = _fuzzy_match_entity_key(candidates, "mylo the golden")
    # "mylo the golden" token_set matches "mylo" (score=100) but not "biscuit" or "luna"
    assert result == "mylo", f"Expected 'mylo' as best match, got {result!r}"


# ── Integration: apply_memories fuzzy dedup ───────────────────────────────────

def test_fuzzy_entity_dedup_merges_variants(tmp_db):
    """Two pet writes with similar keys → only one active row after the second write."""
    with tmp_db.get_db() as conn:
        apply_memories(
            conn,
            user_id="u_fuzzy1",
            source_session_id="s1",
            source_turn_id="t1",
            memories=[_pet("mylo", "cat named Mylo")],
        )
        conn.commit()

    # Second write with a key variant ("mylo the cat" → token_set_ratio=100 vs "mylo")
    with tmp_db.get_db() as conn:
        apply_memories(
            conn,
            user_id="u_fuzzy1",
            source_session_id="s1",
            source_turn_id="t2",
            memories=[_pet("mylo the cat", "cat named Mylo (updated)")],
        )
        conn.commit()

    with tmp_db.get_db() as conn:
        rows = conn.execute(
            "SELECT id, entity_key, value_text, active FROM memories "
            "WHERE user_id='u_fuzzy1' AND slot='pet'"
        ).fetchall()

    active = [r for r in rows if r["active"] == 1]
    assert len(active) == 1, (
        f"Expected 1 active pet row after fuzzy merge, got {len(active)}: "
        f"{[(r['entity_key'], r['value_text']) for r in active]}"
    )
    # Supersession history should be recorded
    inactive = [r for r in rows if r["active"] == 0]
    assert len(inactive) == 1, "Old row should be deactivated (superseded)"


def test_distinct_entities_not_merged(tmp_db):
    """Two pets with distinct names ('mylo' and 'biscuit') remain as two active rows."""
    with tmp_db.get_db() as conn:
        apply_memories(
            conn,
            user_id="u_fuzzy2",
            source_session_id="s2",
            source_turn_id="t1",
            memories=[
                _pet("mylo", "cat named Mylo"),
                _pet("biscuit", "dog named Biscuit"),
            ],
        )
        conn.commit()

    with tmp_db.get_db() as conn:
        rows = conn.execute(
            "SELECT entity_key, active FROM memories "
            "WHERE user_id='u_fuzzy2' AND slot='pet' AND active=1"
        ).fetchall()

    keys = {r["entity_key"] for r in rows}
    assert "mylo" in keys, "Mylo should remain as an active entity"
    assert "biscuit" in keys, "Biscuit should remain as an active entity"
    assert len(rows) == 2, f"Expected 2 distinct pets, got {len(rows)}"


def test_fuzzy_entity_preserves_canonical_key(tmp_db):
    """After a fuzzy merge, the stored entity_key should be the original canonical key."""
    with tmp_db.get_db() as conn:
        apply_memories(
            conn,
            user_id="u_fuzzy3",
            source_session_id="s3",
            source_turn_id="t1",
            memories=[_pet("mylo", "cat named Mylo")],
        )
        conn.commit()

    with tmp_db.get_db() as conn:
        apply_memories(
            conn,
            user_id="u_fuzzy3",
            source_session_id="s3",
            source_turn_id="t2",
            memories=[_pet("mylo the golden", "cat named Mylo (updated)")],
        )
        conn.commit()

    with tmp_db.get_db() as conn:
        active = conn.execute(
            "SELECT entity_key FROM memories WHERE user_id='u_fuzzy3' AND slot='pet' AND active=1"
        ).fetchone()

    # The canonical key "mylo" should be preserved (not replaced by the variant)
    assert active is not None
    assert active["entity_key"] == "mylo", (
        f"Expected canonical key 'mylo', got '{active['entity_key']}'"
    )
