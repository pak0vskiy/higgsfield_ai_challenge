import os
import importlib
import pytest
import asyncio

# Use rules extractor (no LLM key needed in tests)
os.environ["MEMORY_EXTRACTOR"] = "rules"

# Use fake (deterministic) embedder with a small dimension so vec0 wiring
# is fully exercised in tests without any API key.
os.environ["EMBEDDINGS_PROVIDER"] = "fake"
os.environ["EMBEDDING_DIM"] = "64"


@pytest.fixture(scope="function")
def tmp_db(tmp_path):
    """Set DATA_DIR to a temp dir and reload db module. Returns the reloaded db module."""
    os.environ["DATA_DIR"] = str(tmp_path)

    # Reload db module so DB_PATH and VEC_ENABLED are re-evaluated with new DATA_DIR
    import memory_service.db as db_mod
    importlib.reload(db_mod)

    # Also reload embeddings module so EMBEDDING_DIM is picked up
    import memory_service.embeddings as emb_mod
    importlib.reload(emb_mod)

    db_mod.init_db()
    yield db_mod


@pytest.fixture(scope="function")
async def client(tmp_db):
    """Return an async httpx client pointed at the FastAPI app with a fresh DB."""
    from httpx import AsyncClient, ASGITransport
    from memory_service.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
