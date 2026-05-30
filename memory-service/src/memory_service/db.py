import logging
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
DB_PATH = DATA_DIR / "memory.db"

# ── sqlite-vec extension ───────────────────────────────────────────────────────
# Attempt to load the sqlite-vec extension once at module import.
# VEC_ENABLED reflects whether the extension loaded successfully.
# All code that uses the memories_vec virtual table must guard on VEC_ENABLED.
VEC_ENABLED: bool = False
_vec_load_error: str = ""

try:
    import sqlite_vec as _sqlite_vec  # noqa: F401
    VEC_ENABLED = True
except ImportError as _e:
    _vec_load_error = f"sqlite_vec not installed: {_e}"
    logger.warning("sqlite-vec not available — vector recall disabled. %s", _vec_load_error)


def _load_vec_extension(conn: sqlite3.Connection) -> bool:
    """Load the sqlite-vec extension into *conn*. Returns True on success."""
    if not VEC_ENABLED:
        return False
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return True
    except Exception as e:
        logger.warning("Failed to load sqlite-vec extension into connection: %s", e)
        return False


def get_connection() -> sqlite3.Connection:
    """Return a new connection with WAL mode and row_factory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _load_vec_extension(conn)
    return conn


@contextmanager
def get_db():
    """Context manager yielding a connection; closes on exit."""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def _get_embedding_dim() -> int:
    """Read EMBEDDING_DIM from the environment (must match embeddings.py)."""
    try:
        from memory_service.embeddings import EMBEDDING_DIM
        return EMBEDDING_DIM
    except Exception:
        return int(os.getenv("EMBEDDING_DIM", "1536"))


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

        # Create the vector table if sqlite-vec is available.
        # We do this after the main schema so it's safe to call even when
        # VEC_ENABLED is False (the table simply won't exist, and all vec
        # code-paths check VEC_ENABLED before touching it).
        if VEC_ENABLED:
            dim = _get_embedding_dim()
            try:
                conn.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec "
                    f"USING vec0(memory_rowid INTEGER PRIMARY KEY, embedding FLOAT[{dim}])"
                )
            except Exception as e:
                logger.error("Failed to create memories_vec table: %s", e)

        conn.commit()
