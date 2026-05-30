import os, tempfile, pytest, importlib, json

os.environ["MEMORY_EXTRACTOR"] = "rules"

def setup_db(tmp_path):
    os.environ["DATA_DIR"] = str(tmp_path)
    import memory_service.db as db_mod
    importlib.reload(db_mod)
    db_mod.init_db()
    return db_mod

def test_singleton_supersession(tmp_path):
    db_mod = setup_db(tmp_path)
    from memory_service.extractor.base import ExtractedMemory
    from memory_service.memory_engine import apply_memories

    with db_mod.get_db() as conn:
        mem1 = ExtractedMemory(type="fact", slot="location.current", entity_key="", value_text="NYC", confidence=0.95, evidence="I live in NYC", mutation="upsert")
        apply_memories(conn, user_id="u1", source_session_id="s1", source_turn_id="t1", memories=[mem1])
        conn.commit()

    with db_mod.get_db() as conn:
        mem2 = ExtractedMemory(type="fact", slot="location.current", entity_key="", value_text="Berlin", confidence=0.95, evidence="I moved to Berlin", mutation="upsert")
        apply_memories(conn, user_id="u1", source_session_id="s2", source_turn_id="t2", memories=[mem2])
        conn.commit()

    with db_mod.get_db() as conn:
        rows = conn.execute("SELECT slot, value_text, active FROM memories WHERE user_id='u1' ORDER BY created_at").fetchall()
        active_loc = [r for r in rows if r["slot"] == "location.current" and r["active"] == 1]
        inactive_loc = [r for r in rows if r["slot"] == "location.current" and r["active"] == 0]
        assert len(active_loc) == 1
        assert active_loc[0]["value_text"] == "Berlin"
        assert len(inactive_loc) == 1
        assert inactive_loc[0]["value_text"] == "NYC"

def test_previous_chain_auto_written(tmp_path):
    db_mod = setup_db(tmp_path)
    from memory_service.extractor.base import ExtractedMemory
    from memory_service.memory_engine import apply_memories

    with db_mod.get_db() as conn:
        m1 = ExtractedMemory(type="fact", slot="location.current", value_text="NYC", confidence=0.95, evidence="I live in NYC", mutation="upsert")
        apply_memories(conn, user_id="u1", source_session_id="s1", source_turn_id="t1", memories=[m1])
        conn.commit()

    with db_mod.get_db() as conn:
        m2 = ExtractedMemory(type="fact", slot="location.current", value_text="Berlin", confidence=0.95, evidence="moved to Berlin", mutation="upsert")
        apply_memories(conn, user_id="u1", source_session_id="s2", source_turn_id="t2", memories=[m2])
        conn.commit()

    with db_mod.get_db() as conn:
        prev = conn.execute(
            "SELECT value_text FROM memories WHERE user_id='u1' AND slot='location.previous' AND active=1"
        ).fetchone()
        assert prev is not None
        assert prev["value_text"] == "NYC"

def test_collection_multi_entity(tmp_path):
    db_mod = setup_db(tmp_path)
    from memory_service.extractor.base import ExtractedMemory
    from memory_service.memory_engine import apply_memories

    with db_mod.get_db() as conn:
        m1 = ExtractedMemory(type="fact", slot="pet", entity_key="mylo", value_text="cat named Mylo", confidence=0.9, evidence="my cat Mylo", mutation="upsert")
        m2 = ExtractedMemory(type="fact", slot="pet", entity_key="biscuit", value_text="dog named Biscuit", confidence=0.9, evidence="my dog Biscuit", mutation="upsert")
        apply_memories(conn, user_id="u1", source_session_id="s1", source_turn_id="t1", memories=[m1, m2])
        conn.commit()

    with db_mod.get_db() as conn:
        pets = conn.execute("SELECT entity_key FROM memories WHERE user_id='u1' AND slot='pet' AND active=1").fetchall()
        keys = {r["entity_key"] for r in pets}
        assert "mylo" in keys
        assert "biscuit" in keys

def test_low_confidence_stored_but_inactive(tmp_path):
    db_mod = setup_db(tmp_path)
    from memory_service.extractor.base import ExtractedMemory
    from memory_service.memory_engine import apply_memories

    with db_mod.get_db() as conn:
        m = ExtractedMemory(type="fact", slot="location.current", value_text="Maybe Berlin", confidence=0.3, evidence="perhaps Berlin", mutation="upsert")
        apply_memories(conn, user_id="u1", source_session_id="s1", source_turn_id="t1", memories=[m])
        conn.commit()

    with db_mod.get_db() as conn:
        rows = conn.execute("SELECT active FROM memories WHERE user_id='u1' AND slot='location.current'").fetchall()
        assert len(rows) == 1
        assert rows[0]["active"] == 0  # stored but inactive

def test_negate_mutation(tmp_path):
    db_mod = setup_db(tmp_path)
    from memory_service.extractor.base import ExtractedMemory
    from memory_service.memory_engine import apply_memories

    with db_mod.get_db() as conn:
        m1 = ExtractedMemory(type="fact", slot="location.current", value_text="Berlin", confidence=0.9, evidence="I live in Berlin", mutation="upsert")
        apply_memories(conn, user_id="u1", source_session_id="s1", source_turn_id="t1", memories=[m1])
        conn.commit()

    with db_mod.get_db() as conn:
        m2 = ExtractedMemory(type="correction", slot="location.current", entity_key="", value_text="", confidence=0.9, evidence="I don't live there", mutation="negate")
        apply_memories(conn, user_id="u1", source_session_id="s2", source_turn_id="t2", memories=[m2])
        conn.commit()

    with db_mod.get_db() as conn:
        row = conn.execute("SELECT active FROM memories WHERE user_id='u1' AND slot='location.current' AND value_text='Berlin'").fetchone()
        assert row["active"] == 0
