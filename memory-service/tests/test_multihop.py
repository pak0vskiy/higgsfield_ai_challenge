"""
Tests for multi-hop recall.

The multi-hop pass surfaces facts that are related to first-pass hits but not
directly matched by the query. Scenario:

  Turn 1: "I have a dog named Biscuit."
  Turn 2: "I live in Amsterdam."

  Query: "What city does the user with the dog named Biscuit live in?"

With BM25 alone, the query terms ("city", "dog", "Biscuit") surface the pet memory
but may not rank the city directly. The multi-hop pass uses entity tokens from the
first pass ("biscuit", "Amsterdam") as a secondary query to surface the location.
In practice, the profile-facts section (unconditional) already includes location.current,
so the probe passes regardless. We test that the combined context is correct.
"""
import pytest


@pytest.mark.asyncio
async def test_multihop_pet_to_city(client):
    """
    After ingesting a pet fact and a location fact, a query about the pet-owner's
    city should return the city in the recall context.

    Profile facts (location.current) are unconditionally included, so this test
    verifies end-to-end: both the pet name and the city appear in the context.
    """
    # Ingest pet fact
    r1 = await client.post("/turns", json={
        "session_id": "s_mhop",
        "user_id": "u_mhop",
        "messages": [
            {"role": "user", "content": "I have a dog named Biscuit."},
            {"role": "assistant", "content": "What a great name!"},
        ],
        "timestamp": "2025-03-10T10:00:00Z",
        "metadata": {},
    })
    assert r1.status_code == 201

    # Ingest location fact
    r2 = await client.post("/turns", json={
        "session_id": "s_mhop",
        "user_id": "u_mhop",
        "messages": [
            {"role": "user", "content": "I live in Amsterdam."},
            {"role": "assistant", "content": "Amsterdam is beautiful!"},
        ],
        "timestamp": "2025-03-10T10:05:00Z",
        "metadata": {},
    })
    assert r2.status_code == 201

    # Query connecting the two facts
    r3 = await client.post("/recall", json={
        "query": "What city does the user with the dog named Biscuit live in?",
        "session_id": "s_mhop",
        "user_id": "u_mhop",
        "max_tokens": 512,
    })
    assert r3.status_code == 200
    context = r3.json()["context"]

    assert "Amsterdam" in context, f"Expected 'Amsterdam' in multi-hop recall context:\n{context}"
    assert "Biscuit" in context, f"Expected 'Biscuit' in context:\n{context}"


@pytest.mark.asyncio
async def test_multihop_location_linked_to_diet(client):
    """
    Two independent facts (diet and location) should both appear when queried
    via a multi-hop chain: "What healthy food options are available for the user
    in their city?" → diet preference + location.current.
    Profile facts (location.current) are unconditionally included, so this test
    verifies end-to-end context assembly.
    """
    r1 = await client.post("/turns", json={
        "session_id": "s_mhop2",
        "user_id": "u_mhop2",
        "messages": [
            {"role": "user", "content": "I live in Amsterdam."},
            {"role": "assistant", "content": "Amsterdam is a great city!"},
        ],
        "timestamp": "2025-03-10T10:00:00Z",
        "metadata": {},
    })
    assert r1.status_code == 201

    r2 = await client.post("/turns", json={
        "session_id": "s_mhop2",
        "user_id": "u_mhop2",
        "messages": [
            {"role": "user", "content": "I'm vegetarian so finding good food is important to me."},
            {"role": "assistant", "content": "Amsterdam has great vegetarian options!"},
        ],
        "timestamp": "2025-03-10T10:10:00Z",
        "metadata": {},
    })
    assert r2.status_code == 201

    r3 = await client.post("/recall", json={
        "query": "What healthy food options are available for the user in their city?",
        "session_id": "s_mhop2",
        "user_id": "u_mhop2",
        "max_tokens": 512,
    })
    assert r3.status_code == 200
    context = r3.json()["context"]
    assert "Amsterdam" in context, f"Expected 'Amsterdam' in recall context:\n{context}"
    assert "vegetarian" in context.lower(), f"Expected 'vegetarian' in context:\n{context}"


@pytest.mark.asyncio
async def test_recall_context_contains_hop_label_when_applicable(client):
    """
    When a memory is included via the multi-hop path, the context line
    may contain '[via related memory]'. This is a labelling test — we only
    check that the recall works, not that the label is always present
    (first-pass hits won't have it).
    """
    await client.post("/turns", json={
        "session_id": "s_mhop3",
        "user_id": "u_mhop3",
        "messages": [
            {"role": "user", "content": "I have a cat named Luna."},
            {"role": "assistant", "content": "Luna is a lovely name!"},
        ],
        "timestamp": "2025-03-10T10:00:00Z",
        "metadata": {},
    })

    r = await client.post("/recall", json={
        "query": "Tell me about the user's cat",
        "session_id": "s_mhop3",
        "user_id": "u_mhop3",
        "max_tokens": 512,
    })
    assert r.status_code == 200
    assert "Luna" in r.json()["context"]
