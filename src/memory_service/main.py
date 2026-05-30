import json, uuid, logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()  # load .env before any os.getenv() calls

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from memory_service.db import init_db, get_db
from memory_service.schemas import (
    TurnRequest, TurnResponse,
    RecallRequest, RecallResponse,
    SearchRequest, SearchResponse, SearchResult,
    UserMemoriesResponse, MemoryRecord,
)
from memory_service.extractor.llm import get_extractor
from memory_service.memory_engine import apply_memories, get_user_context_summary
from memory_service.recall import recall, search
from memory_service.auth import AuthMiddleware

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="Memory Service", version="0.1.0", lifespan=lifespan)
app.add_middleware(AuthMiddleware)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url, exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/turns", status_code=201)
async def post_turns(body: TurnRequest) -> TurnResponse:
    turn_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    user_id = body.user_id or f"anon:{body.session_id}"

    messages = [m.model_dump() for m in body.messages]

    with get_db() as conn:
        # 1. Store raw turn
        conn.execute(
            """INSERT INTO turns (id, session_id, user_id, messages_json, timestamp, metadata_json, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (turn_id, body.session_id, user_id,
             json.dumps(messages),
             body.timestamp,
             json.dumps(body.metadata),
             now)
        )
        conn.commit()

        # 2. Get user context for LLM extraction
        user_context = get_user_context_summary(conn, user_id)

        # 3. Extract memories
        extractor = get_extractor()
        try:
            result = extractor.extract(messages, user_context=user_context)
        except Exception as e:
            logger.warning("Extraction failed for turn %s: %s", turn_id, e)
            result = None

        # 4. Apply extracted memories
        if result and result.memories:
            apply_memories(
                conn,
                user_id=user_id,
                source_session_id=body.session_id,
                source_turn_id=turn_id,
                memories=result.memories,
            )
            conn.commit()

    return TurnResponse(id=turn_id)


@app.post("/recall")
async def post_recall(body: RecallRequest) -> RecallResponse:
    with get_db() as conn:
        result = recall(
            conn,
            query=body.query,
            session_id=body.session_id,
            user_id=body.user_id,
            max_tokens=body.max_tokens,
        )
    from memory_service.schemas import Citation
    citations = [Citation(**c) for c in result["citations"]]
    return RecallResponse(context=result["context"], citations=citations)


@app.post("/search")
async def post_search(body: SearchRequest) -> SearchResponse:
    with get_db() as conn:
        results = search(
            conn,
            query=body.query,
            session_id=body.session_id,
            user_id=body.user_id,
            limit=body.limit,
        )
    items = [SearchResult(**r) for r in results]
    return SearchResponse(results=items)


@app.get("/users/{user_id}/memories")
async def get_user_memories(user_id: str) -> UserMemoriesResponse:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT id, type, slot, entity_key, value_text, confidence,
                      source_session_id, source_turn_id, created_at, updated_at,
                      supersedes, active, evidence
               FROM memories
               WHERE user_id=?
               ORDER BY updated_at DESC""",
            (user_id,)
        ).fetchall()

    memories = []
    for r in rows:
        memories.append(MemoryRecord(
            id=r["id"],
            type=r["type"],
            key=r["slot"],          # 'key' in API = 'slot' internally
            value=r["value_text"],
            confidence=r["confidence"],
            source_session=r["source_session_id"] or "",
            source_turn=r["source_turn_id"] or "",
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            supersedes=r["supersedes"],
            active=bool(r["active"]),
            slot=r["slot"],
            entity_key=r["entity_key"] or "",
            evidence=r["evidence"] or "",
        ))

    return UserMemoriesResponse(memories=memories)


@app.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str):
    """Delete raw turns for session. User-level memories remain (they're user-scoped)."""
    with get_db() as conn:
        conn.execute("DELETE FROM turns WHERE session_id=?", (session_id,))
        conn.commit()


@app.delete("/users/{user_id}", status_code=204)
async def delete_user(user_id: str):
    """Delete all data for a user."""
    with get_db() as conn:
        conn.execute("DELETE FROM memories WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM turns WHERE user_id=?", (user_id,))
        conn.commit()
