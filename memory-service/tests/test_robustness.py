import pytest

pytestmark = pytest.mark.asyncio


async def test_bad_json_returns_422(client):
    # Send raw bytes — not valid JSON
    r = await client.post("/turns", content=b"not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


async def test_missing_required_fields_returns_422(client):
    r = await client.post("/turns", json={"session_id": "s1"})  # missing messages, timestamp
    assert r.status_code == 422


async def test_empty_body_returns_422(client):
    r = await client.post("/turns", json={})
    assert r.status_code == 422


async def test_unicode_in_content_doesnt_crash(client):
    r = await client.post("/turns", json={
        "session_id": "s1", "user_id": "u1",
        "messages": [{"role": "user", "content": "こんにちは! I live in 東京. 🐱"}],
        "timestamp": "2025-03-15T10:00:00Z", "metadata": {}
    })
    assert r.status_code == 201


async def test_large_content_doesnt_crash(client):
    big_content = "word " * 5000
    r = await client.post("/turns", json={
        "session_id": "s1", "user_id": "u1",
        "messages": [{"role": "user", "content": big_content}],
        "timestamp": "2025-03-15T10:00:00Z", "metadata": {}
    })
    assert r.status_code in (201, 413)


async def test_recall_on_cold_session_returns_empty(client):
    r = await client.post("/recall", json={
        "query": "Where does this user live?",
        "session_id": "new-session-xyz",
        "user_id": "new-user-xyz",
        "max_tokens": 512
    })
    assert r.status_code == 200
    data = r.json()
    assert data["context"] == ""
    assert data["citations"] == []


async def test_search_with_no_results_returns_empty(client):
    r = await client.post("/search", json={
        "query": "zyxwvutsrq_never_mentioned",
        "user_id": "u-nonexistent",
        "limit": 10
    })
    assert r.status_code == 200
    assert r.json()["results"] == []


async def test_multi_message_turn_handled(client):
    r = await client.post("/turns", json={
        "session_id": "s1", "user_id": "u1",
        "messages": [
            {"role": "user", "content": "Can you help with my React app?"},
            {"role": "assistant", "content": "Sure! What's the issue?"},
            {"role": "tool", "name": "search", "content": "React documentation"},
            {"role": "user", "content": "I work at Notion as an engineer."},
        ],
        "timestamp": "2025-03-15T10:00:00Z", "metadata": {}
    })
    assert r.status_code == 201
