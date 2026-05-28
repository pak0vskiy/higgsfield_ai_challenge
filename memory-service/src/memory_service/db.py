import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
DB_PATH = DATA_DIR / "memory.db"


def get_connection() -> sqlite3.Connection:
    """Return a new connection with WAL mode and row_factory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    """Context manager yielding a connection; closes on exit."""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Run migrations — idempotent, safe to call on every startup."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS turns (
                id           TEXT PRIMARY KEY,
                session_id   TEXT NOT NULL,
                user_id      TEXT,
                messages_json TEXT NOT NULL,
                timestamp    TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at   TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_turns_session
                ON turns(session_id);

            CREATE INDEX IF NOT EXISTS idx_turns_user
                ON turns(user_id);

            CREATE TABLE IF NOT EXISTS memories (
                id               TEXT PRIMARY KEY,
                user_id          TEXT NOT NULL,
                source_session_id TEXT,
                source_turn_id   TEXT,
                type             TEXT NOT NULL,
                slot             TEXT NOT NULL,
                entity_key       TEXT NOT NULL DEFAULT '',
                value_text       TEXT NOT NULL,
                attributes_json  TEXT NOT NULL DEFAULT '{}',
                confidence       REAL NOT NULL DEFAULT 1.0,
                evidence         TEXT NOT NULL DEFAULT '',
                active           INTEGER NOT NULL DEFAULT 1,
                supersedes       TEXT,
                created_at       TEXT NOT NULL,
                updated_at       TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_mem_lookup
                ON memories(user_id, slot, entity_key, active);

            CREATE INDEX IF NOT EXISTS idx_mem_user_active
                ON memories(user_id, active);

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                value_text,
                evidence,
                slot,
                content='memories',
                content_rowid='rowid'
            );

            -- FTS5 sync triggers
            CREATE TRIGGER IF NOT EXISTS memories_ai
                AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, value_text, evidence, slot)
                    VALUES (new.rowid, new.value_text, new.evidence, new.slot);
                END;

            CREATE TRIGGER IF NOT EXISTS memories_ad
                AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, value_text, evidence, slot)
                    VALUES ('delete', old.rowid, old.value_text, old.evidence, old.slot);
                END;

            CREATE TRIGGER IF NOT EXISTS memories_au
                AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, value_text, evidence, slot)
                    VALUES ('delete', old.rowid, old.value_text, old.evidence, old.slot);
                    INSERT INTO memories_fts(rowid, value_text, evidence, slot)
                    VALUES (new.rowid, new.value_text, new.evidence, new.slot);
                END;
        """)
        conn.commit()
