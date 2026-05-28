import os
import importlib
import pytest
import asyncio

os.environ["MEMORY_EXTRACTOR"] = "rules"


@pytest.fixture(scope="function")
def tmp_db(tmp_path):
    """Set DATA_DIR to a temp dir and reload db module. Returns the reloaded db module."""
    os.environ["DATA_DIR"] = str(tmp_path)
    import memory_service.db as db_mod
    importlib.reload(db_mod)
    db_mod.init_db()
    yield db_mod


@pytest.fixture(scope="function")
async def client(tmp_db):
    """Return an async httpx client pointed at the FastAPI app with a fresh DB."""
    from httpx import AsyncClient, ASGITransport
    from memory_service.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
