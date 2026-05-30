import importlib
import os

import pytest


def test_init_db(tmp_path):
    os.environ["DATA_DIR"] = str(tmp_path)
    # reimport to pick up new DATA_DIR
    import memory_service.db as db_mod
    importlib.reload(db_mod)
    db_mod.init_db()
    with db_mod.get_db() as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "turns" in tables
        assert "memories" in tables
        # FTS5 presence check
        fts_tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE name LIKE 'memories_fts%'")}
        assert len(fts_tables) > 0, "memories_fts virtual table not created"
