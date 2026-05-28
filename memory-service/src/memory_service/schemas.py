from typing import Any, Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class MessageIn(BaseModel):
    role: str  # "user" | "assistant" | "tool"
    content: str
    name: Optional[str] = None  # for tool messages


class TurnRequest(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    messages: list[MessageIn]
    timestamp: str  # ISO-8601
    metadata: dict[str, Any] = {}


class RecallRequest(BaseModel):
    query: str
    session_id: str
    user_id: Optional[str] = None
    max_tokens: int = 1024


class SearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    limit: int = 10


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class TurnResponse(BaseModel):
    id: str


class Citation(BaseModel):
    turn_id: str
    score: float
    snippet: str


class RecallResponse(BaseModel):
    context: str
    citations: list[Citation]


class SearchResult(BaseModel):
    content: str
    score: float
    session_id: str
    timestamp: str
    metadata: dict[str, Any] = {}


class SearchResponse(BaseModel):
    results: list[SearchResult]


class MemoryRecord(BaseModel):
    id: str
    type: str                       # fact | preference | opinion | event | correction
    key: str                        # = slot (for API compatibility with task spec)
    value: str
    confidence: float
    source_session: str
    source_turn: str
    created_at: str
    updated_at: str
    supersedes: Optional[str] = None
    active: bool
    # extras not in task spec but useful for inspection
    slot: str
    entity_key: str
    evidence: str


class UserMemoriesResponse(BaseModel):
    memories: list[MemoryRecord]
