import pytest
from httpx import AsyncClient

# Use the client fixture from conftest

pytestmark = pytest.mark.asyncio

TURN_PAYLOAD = {
    "session_id": "s1",
    "user_id": "u1",
    "messages": [
        {"role": "user", "content": "I just moved to Berlin from NYC last month."},
        {"role": "assistant", "content": "Berlin is a great city!"}
    ],
    "timestamp": "2025-03-15T10:30:00Z",
    "metadata": {}
}


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_post_turns_creates_turn(client):
    r = await client.post("/turns", json=TURN_PAYLOAD)
    assert r.status_code == 201
    data = r.json()
    assert "id" in data
    assert isinstance(data["id"], str)


async def test_recall_returns_correct_shape(client):
    await client.post("/turns", json=TURN_PAYLOAD)
    r = await client.post("/recall", json={
        "query": "Where does the user live?",
        "session_id": "s2",
        "user_id": "u1",
        "max_tokens": 512
    })
    assert r.status_code == 200
    data = r.json()
    assert "context" in data
    assert "citations" in data
    assert isinstance(data["citations"], list)


async def test_recall_mentions_berlin_after_turn(client):
    await client.post("/turns", json=TURN_PAYLOAD)
    r = await client.post("/recall", json={
        "query": "Where does the user live?",
        "session_id": "s2",
        "user_id": "u1",
        "max_tokens": 512
    })
    assert "Berlin" in r.json()["context"]


async def test_search_returns_correct_shape(client):
    await client.post("/turns", json=TURN_PAYLOAD)
    r = await client.post("/search", json={
        "query": "Berlin",
        "user_id": "u1",
        "limit": 5
    })
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    for item in data["results"]:
        assert "content" in item
        assert "score" in item
        assert "session_id" in item
        assert "timestamp" in item


async def test_user_memories_returns_structured_data(client):
    await client.post("/turns", json=TURN_PAYLOAD)
    r = await client.get("/users/u1/memories")
    assert r.status_code == 200
    data = r.json()
    assert "memories" in data
    assert len(data["memories"]) > 0
    mem = data["memories"][0]
    assert "id" in mem
    assert "type" in mem
    assert "key" in mem
    assert "value" in mem
    assert "active" in mem


async def test_delete_session_returns_204(client):
    await client.post("/turns", json=TURN_PAYLOAD)
    r = await client.delete("/sessions/s1")
    assert r.status_code == 204


async def test_delete_user_returns_204(client):
    await client.post("/turns", json=TURN_PAYLOAD)
    r = await client.delete("/users/u1")
    assert r.status_code == 204


async def test_delete_session_does_not_remove_memories(client):
    """Deleting a session removes turns but leaves user-level memories."""
    await client.post("/turns", json=TURN_PAYLOAD)
    # Memories exist before delete
    r1 = await client.get("/users/u1/memories")
    assert len(r1.json()["memories"]) > 0
    # Delete the session
    await client.delete("/sessions/s1")
    # Memories still exist
    r2 = await client.get("/users/u1/memories")
    assert len(r2.json()["memories"]) > 0


async def test_delete_user_removes_everything(client):
    await client.post("/turns", json=TURN_PAYLOAD)
    await client.delete("/users/u1")
    r = await client.get("/users/u1/memories")
    assert r.json()["memories"] == []
