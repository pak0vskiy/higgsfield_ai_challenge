"""
Tests for bearer token authentication middleware.

When MEMORY_AUTH_TOKEN is set, all endpoints except /health require
Authorization: Bearer <token>. When it is unset/empty, auth is disabled.
"""
import os
import pytest


@pytest.fixture(autouse=True)
def clear_auth_token():
    """Ensure MEMORY_AUTH_TOKEN is unset before and after each test."""
    os.environ.pop("MEMORY_AUTH_TOKEN", None)
    yield
    os.environ.pop("MEMORY_AUTH_TOKEN", None)


@pytest.mark.asyncio
async def test_no_auth_required_when_token_unset(client):
    r = await client.get("/health")
    assert r.status_code == 200

    r = await client.post("/recall", json={
        "query": "anything",
        "session_id": "s1",
        "user_id": "u1",
        "max_tokens": 512,
    })
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_health_exempt_even_when_token_set(client):
    os.environ["MEMORY_AUTH_TOKEN"] = "secret123"
    r = await client.get("/health")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_missing_token_returns_401(client):
    os.environ["MEMORY_AUTH_TOKEN"] = "secret123"
    r = await client.post("/recall", json={
        "query": "anything",
        "session_id": "s1",
        "user_id": "u1",
        "max_tokens": 512,
    })
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_wrong_token_returns_403(client):
    os.environ["MEMORY_AUTH_TOKEN"] = "secret123"
    r = await client.post("/recall", json={
        "query": "anything",
        "session_id": "s1",
        "user_id": "u1",
        "max_tokens": 512,
    }, headers={"Authorization": "Bearer wrongtoken"})
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_correct_token_allows_request(client):
    os.environ["MEMORY_AUTH_TOKEN"] = "secret123"
    r = await client.post("/recall", json={
        "query": "anything",
        "session_id": "s1",
        "user_id": "u1",
        "max_tokens": 512,
    }, headers={"Authorization": "Bearer secret123"})
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_turns_endpoint_also_requires_auth(client):
    os.environ["MEMORY_AUTH_TOKEN"] = "secret123"
    r = await client.post("/turns", json={
        "session_id": "s1",
        "user_id": "u1",
        "messages": [{"role": "user", "content": "hello"}],
        "timestamp": "2025-03-10T10:00:00Z",
        "metadata": {},
    })
    assert r.status_code == 401
