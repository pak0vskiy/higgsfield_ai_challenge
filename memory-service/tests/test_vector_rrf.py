"""
Tests for the vector retrieval channel and RRF fusion.

Uses the fake embedder (set in conftest.py — EMBEDDINGS_PROVIDER=fake, EMBEDDING_DIM=64).
sqlite-vec is expected to be available in the test environment; if it isn't, vector
tests are skipped gracefully.
"""
import pytest
from memory_service.extractor.base import ExtractedMemory


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_mem(slot="location.current", value="Berlin", entity_key="", confidence=0.9):
    return ExtractedMemory(
        type="fact",
        slot=slot,
        entity_key=entity_key,
        value_text=value,
        confidence=confidence,
        evidence=value,
        mutation="upsert",
    )


def _insert_and_commit(tmp_db, user_id, session_id, turn_id, mem):
    from memory_service.memory_engine import apply_memories
    with tmp_db.get_db() as conn:
        apply_memories(
            conn,
            user_id=user_id,
            source_session_id=session_id,
            source_turn_id=turn_id,
            memories=[mem],
        )
        conn.commit()


def _vec_enabled(tmp_db):
    return tmp_db.VEC_ENABLED


# ── Vector row written on insert ───────────────────────────────────────────────

def test_vec_row_written_on_memory_insert(tmp_db):
    if not _vec_enabled(tmp_db):
        pytest.skip("sqlite-vec not available")

    _insert_and_commit(tmp_db, "user1", "s1", "t1", _make_mem("location.current", "Berlin"))

    with tmp_db.get_db() as conn:
        # The memories_vec table should have at least one row
        count = conn.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]
        assert count >= 1, "Vector row should be written for each inserted memory"


def test_multiple_memories_each_get_vec_row(tmp_db):
    if not _vec_enabled(tmp_db):
        pytest.skip("sqlite-vec not available")

    mems = [
        _make_mem("location.current", "Berlin"),
        _make_mem("preference.diet", "vegetarian"),
        _make_mem("pet", "dog named Biscuit", entity_key="biscuit"),
    ]
    with tmp_db.get_db() as conn:
        from memory_service.memory_engine import apply_memories
        apply_memories(conn, user_id="user2", source_session_id="s2", source_turn_id="t2", memories=mems)
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]
        assert count >= 3


# ── RRF fusion ─────────────────────────────────────────────────────────────────

def test_rrf_fuse_both_channels():
    from memory_service.recall import _rrf_fuse

    bm25 = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    vec  = [{"id": "b"}, {"id": "a"}, {"id": "d"}]

    scores = _rrf_fuse(bm25, vec)

    # "b" appears in both channels → should score higher than "c" (BM25 only) and "d" (vec only)
    assert scores["b"] > scores["c"], "Double-channel hit should outscore single-channel"
    assert scores["b"] > scores["d"], "Double-channel hit should outscore single-channel"


def test_rrf_fuse_single_channel_degrades_gracefully():
    from memory_service.recall import _rrf_fuse

    bm25 = [{"id": "x"}, {"id": "y"}]
    vec  = []

    scores = _rrf_fuse(bm25, vec)
    assert "x" in scores
    assert "y" in scores
    assert scores["x"] > scores["y"], "First-ranked should score higher in single-channel RRF"


def test_rrf_fuse_empty_both_channels():
    from memory_service.recall import _rrf_fuse
    assert _rrf_fuse([], []) == {}


# ── Vector search ──────────────────────────────────────────────────────────────

def test_vector_search_returns_inserted_memory(tmp_db):
    if not _vec_enabled(tmp_db):
        pytest.skip("sqlite-vec not available")

    _insert_and_commit(tmp_db, "user3", "s3", "t3", _make_mem("location.current", "Tokyo"))

    with tmp_db.get_db() as conn:
        from memory_service.recall import _vector_search
        results = _vector_search(conn, "user3", "Where does the user live?", limit=5)
        # At least one result should be returned for this user
        assert any(r["value_text"] == "Tokyo" for r in results), \
            f"Expected 'Tokyo' in vector search results, got: {results}"


def test_vector_search_respects_user_isolation(tmp_db):
    if not _vec_enabled(tmp_db):
        pytest.skip("sqlite-vec not available")

    _insert_and_commit(tmp_db, "user_a", "s_a", "t_a", _make_mem("location.current", "Paris"))
    _insert_and_commit(tmp_db, "user_b", "s_b", "t_b", _make_mem("location.current", "Madrid"))

    with tmp_db.get_db() as conn:
        from memory_service.recall import _vector_search
        results_a = _vector_search(conn, "user_a", "location", limit=10)
        results_b = _vector_search(conn, "user_b", "location", limit=10)

    ids_a = {r["id"] for r in results_a}
    ids_b = {r["id"] for r in results_b}
    assert ids_a.isdisjoint(ids_b), "Vector search must not cross user boundaries"


def test_vector_search_returns_empty_when_vec_disabled(tmp_db, monkeypatch):
    """When VEC_ENABLED is False, _vector_search must return []."""
    monkeypatch.setattr("memory_service.db.VEC_ENABLED", False)
    with tmp_db.get_db() as conn:
        from memory_service.recall import _vector_search
        results = _vector_search(conn, "user_x", "anything", limit=5)
    assert results == []


# ── Recall degrades to BM25 when vec is off ────────────────────────────────────

@pytest.mark.asyncio
async def test_recall_works_bm25_only_when_vec_disabled(tmp_db, monkeypatch, client):
    """Even with VEC_ENABLED=False, recall should still return BM25-based results."""
    monkeypatch.setattr("memory_service.db.VEC_ENABLED", False)

    r = await client.post("/turns", json={
        "session_id": "s_bm25",
        "user_id": "u_bm25",
        "messages": [
            {"role": "user", "content": "I live in Amsterdam."},
            {"role": "assistant", "content": "Nice city!"},
        ],
        "timestamp": "2025-03-10T10:00:00Z",
        "metadata": {},
    })
    assert r.status_code == 201

    r2 = await client.post("/recall", json={
        "query": "Where does the user live?",
        "session_id": "s_bm25",
        "user_id": "u_bm25",
        "max_tokens": 512,
    })
    assert r2.status_code == 200
    context = r2.json()["context"]
    assert "Amsterdam" in context, f"Expected Amsterdam in BM25-only recall, got: {context}"
