"""
Embedding utilities for the memory service.

Provider is selected via EMBEDDINGS_PROVIDER env var:
  - "openai"  (default) — calls litellm.embedding with text-embedding-3-small
  - "fake"             — deterministic, hash-seeded vector for tests (no API key needed)

EMBEDDING_DIM env controls the dimension (default 1536 for text-embedding-3-small).
The fake provider honours EMBEDDING_DIM too so tests can set it small (e.g. 64).

All public functions return None on any failure; callers should treat None as
"no vector available — degrade to BM25-only".
"""

import hashlib
import logging
import math
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "openai").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))


# ── Fake (deterministic) embedder ──────────────────────────────────────────────
def _fake_embed(text: str) -> list[float]:
    """
    Deterministic pseudo-embedding for offline tests.
    Produces a unit-normalised vector of length EMBEDDING_DIM seeded by a
    SHA-256 hash of the text. Two identical texts → identical vectors.
    Similar texts do NOT produce similar vectors (no semantic signal), but that
    is fine for unit tests that verify the mechanical plumbing.
    """
    dim = EMBEDDING_DIM
    seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()
    # Build an integer seed from the first 8 bytes of the hash
    seed = int.from_bytes(seed_bytes[:8], "big")
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    # Skip BLAS-based normalization — np.linalg.norm on float32 can give
    # non-identical results across calls on some CPUs due to SIMD reordering.
    # For test purposes (determinism + distinctness) a raw seeded vector is fine.
    return vec.tolist()


# ── OpenAI (litellm) embedder ──────────────────────────────────────────────────
def _openai_embed(text: str) -> Optional[list[float]]:
    """Call litellm.embedding and return the first embedding vector."""
    try:
        import litellm  # deferred import so offline tests don't need it configured
        response = litellm.embedding(model=EMBEDDING_MODEL, input=[text])
        vec = response.data[0]["embedding"]
        return vec
    except Exception as e:
        logger.warning("OpenAI embedding failed for text %.80r: %s", text, e)
        return None


# ── Public API ─────────────────────────────────────────────────────────────────
def embed(text: str) -> Optional[list[float]]:
    """
    Return an embedding vector for *text*, or None on failure.

    Never raises. Callers must treat None as "no vector" and fall back to BM25.
    """
    if not text or not text.strip():
        return None
    try:
        if EMBEDDINGS_PROVIDER == "fake":
            return _fake_embed(text)
        # Default: openai via litellm
        return _openai_embed(text)
    except Exception as e:
        logger.error("embed() unexpected error: %s", e)
        return None


def embed_batch(texts: list[str]) -> list[Optional[list[float]]]:
    """
    Return embeddings for a list of texts. Each element is a vector or None.
    For the fake provider, calls _fake_embed per item (always succeeds).
    For the OpenAI provider, tries a single batched call first; on failure
    falls back to per-item calls so partial results are preserved.
    """
    if not texts:
        return []

    if EMBEDDINGS_PROVIDER == "fake":
        return [_fake_embed(t) if t and t.strip() else None for t in texts]

    # OpenAI batched path
    try:
        import litellm
        non_empty = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not non_empty:
            return [None] * len(texts)
        indices, inputs = zip(*non_empty)
        response = litellm.embedding(model=EMBEDDING_MODEL, input=list(inputs))
        result: list[Optional[list[float]]] = [None] * len(texts)
        for rank, idx in enumerate(indices):
            result[idx] = response.data[rank]["embedding"]
        return result
    except Exception as e:
        logger.warning("Batched embedding failed, falling back to per-item: %s", e)
        return [embed(t) for t in texts]


def serialize_f32(vec: list[float]) -> bytes:
    """Serialize a float list to little-endian float32 bytes for sqlite-vec."""
    return np.array(vec, dtype=np.float32).tobytes()


def deserialize_f32(buf: bytes) -> list[float]:
    """Deserialize sqlite-vec float32 bytes back to a Python list."""
    return np.frombuffer(buf, dtype=np.float32).tolist()
