import pytest

pytestmark = pytest.mark.asyncio


async def test_different_users_dont_bleed(client):
    """u1 and u2 memories are completely isolated."""
    await client.post("/turns", json={
        "session_id": "s1", "user_id": "u1",
        "messages": [{"role": "user", "content": "I live in Berlin."}],
        "timestamp": "2025-03-15T10:00:00Z", "metadata": {}
    })
    await client.post("/turns", json={
        "session_id": "s2", "user_id": "u2",
        "messages": [{"role": "user", "content": "I live in Tokyo."}],
        "timestamp": "2025-03-15T10:01:00Z", "metadata": {}
    })
    r1 = await client.post("/recall", json={"query": "Where does the user live?", "session_id": "s1", "user_id": "u1", "max_tokens": 512})
    r2 = await client.post("/recall", json={"query": "Where does the user live?", "session_id": "s2", "user_id": "u2", "max_tokens": 512})

    assert "Berlin" in r1.json()["context"]
    assert "Tokyo" not in r1.json()["context"]
    assert "Tokyo" in r2.json()["context"]
    assert "Berlin" not in r2.json()["context"]


async def test_same_user_memories_shared_across_sessions(client):
    """u1's memories from s1 are visible when recalling from s2."""
    await client.post("/turns", json={
        "session_id": "s1", "user_id": "u1",
        "messages": [{"role": "user", "content": "I live in Berlin."}],
        "timestamp": "2025-03-15T10:00:00Z", "metadata": {}
    })
    r = await client.post("/recall", json={
        "query": "Where does this user live?",
        "session_id": "s2",  # different session
        "user_id": "u1",
        "max_tokens": 512
    })
    assert "Berlin" in r.json()["context"]


async def test_null_user_id_isolated_by_session(client):
    """Turns with null user_id get scoped to anon:{session_id}."""
    await client.post("/turns", json={
        "session_id": "s-anon",
        "user_id": None,
        "messages": [{"role": "user", "content": "I live in Vienna."}],
        "timestamp": "2025-03-15T10:00:00Z", "metadata": {}
    })
    r = await client.post("/recall", json={
        "query": "Where does the user live?",
        "session_id": "s-anon",
        "user_id": None,
        "max_tokens": 512
    })
    assert "Vienna" in r.json()["context"]
