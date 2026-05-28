import os, importlib, json, pytest

os.environ["MEMORY_EXTRACTOR"] = "rules"

def setup_env(tmp_path):
    os.environ["DATA_DIR"] = str(tmp_path)
    import memory_service.db as db_mod
    importlib.reload(db_mod)
    db_mod.init_db()
    return db_mod

def insert_turn(conn, session_id, user_id, content, ts="2025-03-15T10:00:00Z"):
    import uuid
    from datetime import datetime, timezone
    tid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    msgs = json.dumps([{"role": "user", "content": content}])
    conn.execute(
        "INSERT INTO turns (id, session_id, user_id, messages_json, timestamp, metadata_json, created_at) VALUES (?,?,?,?,?,?,?)",
        (tid, session_id, user_id, msgs, ts, "{}", now)
    )
    conn.commit()
    return tid

def insert_memory(conn, user_id, slot, value, entity_key="", confidence=0.9, turn_id="t1", session_id="s1"):
    import uuid
    from datetime import datetime, timezone
    mid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO memories (id, user_id, source_session_id, source_turn_id, type, slot, entity_key,
           value_text, attributes_json, confidence, evidence, active, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (mid, user_id, session_id, turn_id, "fact", slot, entity_key,
         value, "{}", confidence, value, 1, now, now)
    )
    conn.commit()
    return mid

def test_recall_returns_profile_facts(tmp_path):
    db_mod = setup_env(tmp_path)
    from memory_service.recall import recall

    with db_mod.get_db() as conn:
        insert_memory(conn, "u1", "location.current", "Berlin")
        result = recall(conn, query="Where does the user live?", session_id="s1", user_id="u1", max_tokens=512)

    assert "Berlin" in result["context"]
    assert result["citations"]

def test_recall_cold_session_returns_empty(tmp_path):
    db_mod = setup_env(tmp_path)
    from memory_service.recall import recall

    with db_mod.get_db() as conn:
        result = recall(conn, query="Where does the user live?", session_id="new-session", user_id="never-seen", max_tokens=512)

    assert result["context"] == ""
    assert result["citations"] == []

def test_recall_respects_user_isolation(tmp_path):
    db_mod = setup_env(tmp_path)
    from memory_service.recall import recall

    with db_mod.get_db() as conn:
        insert_memory(conn, "u1", "location.current", "Berlin")
        insert_memory(conn, "u2", "location.current", "Tokyo")

        result_u1 = recall(conn, query="Where does the user live?", session_id="s1", user_id="u1", max_tokens=512)
        result_u2 = recall(conn, query="Where does the user live?", session_id="s2", user_id="u2", max_tokens=512)

    assert "Berlin" in result_u1["context"]
    assert "Tokyo" not in result_u1["context"]
    assert "Tokyo" in result_u2["context"]
    assert "Berlin" not in result_u2["context"]

def test_recall_respects_max_tokens_soft_cap(tmp_path):
    db_mod = setup_env(tmp_path)
    from memory_service.recall import recall

    with db_mod.get_db() as conn:
        # Insert many memories to trigger budget
        for i in range(50):
            insert_memory(conn, "u1", "unstructured", f"Memory fact number {i} about something interesting " * 5)
        result = recall(conn, query="memory fact", session_id="s1", user_id="u1", max_tokens=100)

    # Should not exceed 2x the max_tokens drastically
    from memory_service.recall import _count_tokens
    token_count = _count_tokens(result["context"])
    assert token_count <= 200  # 2x soft cap

def test_search_returns_structured_results(tmp_path):
    db_mod = setup_env(tmp_path)
    from memory_service.recall import search

    with db_mod.get_db() as conn:
        insert_memory(conn, "u1", "location.current", "Berlin")
        results = search(conn, query="Berlin", session_id=None, user_id="u1", limit=10)

    assert len(results) >= 1
    assert results[0]["content"] == "Berlin"
    assert "score" in results[0]
    assert "timestamp" in results[0]
