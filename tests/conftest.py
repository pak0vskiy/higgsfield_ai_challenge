import os
import importlib
import pytest
import asyncio
from dotenv import load_dotenv

# Load .env first so its values are in os.environ before setdefault calls below.
# load_dotenv() does not override vars already set in the OS environment.
load_dotenv()

# Use rules extractor by default (no LLM key needed).
# Override by setting MEMORY_EXTRACTOR=llm in your environment or .env.
os.environ.setdefault("MEMORY_EXTRACTOR", "rules")

# Use fake (deterministic) embedder with a small dimension by default so
# vec0 wiring is exercised without any API key.
# Override by setting EMBEDDINGS_PROVIDER=openai (requires OPENAI_API_KEY).
os.environ.setdefault("EMBEDDINGS_PROVIDER", "fake")
# EMBEDDING_DIM is now provider-aware in embeddings.py (fake=64, openai=1536).
# No need to set it here — the module picks the right default automatically.


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
