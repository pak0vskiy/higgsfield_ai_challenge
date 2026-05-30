from pydantic import BaseModel, Field
from typing import Literal, Optional, Any

MemoryType = Literal["fact", "preference", "opinion", "event", "correction"]
MutationIntent = Literal["upsert", "replace", "append", "negate"]


class ExtractedMemory(BaseModel):
    type: MemoryType
    slot: str                           # canonical slot or "unstructured"
    entity_key: str = ""               # empty for singletons; normalized name for collections
    value_text: str                     # human-readable value
    attributes: dict[str, Any] = {}    # extra structured attributes
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: str = ""                  # verbatim span from conversation
    mutation: MutationIntent = "upsert"
    supersedes_hint: Optional[str] = None  # LLM hint about what value this replaces


class ExtractionResult(BaseModel):
    memories: list[ExtractedMemory]
    extractor_used: str                 # "llm" | "rules"
    raw_llm_response: Optional[str] = None
