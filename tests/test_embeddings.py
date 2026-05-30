"""
Tests for embeddings.py.

All tests use the fake provider (set in conftest.py) so no API key is needed.
"""
import os
import pytest
import importlib


def _get_embeddings_module(provider: str = "fake"):
    """Reload embeddings module with the given provider. Defaults to 'fake'."""
    os.environ["EMBEDDINGS_PROVIDER"] = provider
    os.environ.setdefault("EMBEDDING_DIM", "64")
    import memory_service.embeddings as m
    importlib.reload(m)
    return m


def test_fake_embed_returns_correct_dimension():
    emb = _get_embeddings_module()
    vec = emb.embed("Hello, world!")
    assert vec is not None
    assert len(vec) == int(os.environ.get("EMBEDDING_DIM", "64"))


def test_fake_embed_is_deterministic():
    emb = _get_embeddings_module()
    text = "I live in Berlin and work at Acme Corp"
    v1 = emb.embed(text)
    v2 = emb.embed(text)
    assert v1 is not None
    assert v1 == v2, "Fake embedder must return identical vectors for the same input"


def test_fake_embed_differs_across_texts():
    emb = _get_embeddings_module()
    v1 = emb.embed("I live in Berlin")
    v2 = emb.embed("I have a dog named Biscuit")
    assert v1 is not None
    assert v2 is not None
    assert v1 != v2, "Different texts should produce different fake vectors"


def test_embed_returns_none_for_empty_string():
    emb = _get_embeddings_module()
    assert emb.embed("") is None
    assert emb.embed("   ") is None


def test_embed_batch_returns_list_of_same_length():
    emb = _get_embeddings_module()
    texts = ["apple", "banana", "cherry"]
    results = emb.embed_batch(texts)
    assert len(results) == len(texts)
    for vec in results:
        assert vec is not None
        assert len(vec) == int(os.environ.get("EMBEDDING_DIM", "64"))


def test_embed_batch_handles_empty_strings_in_list():
    emb = _get_embeddings_module()
    results = emb.embed_batch(["hello", "", "world"])
    assert results[0] is not None
    assert results[1] is None
    assert results[2] is not None


def test_embed_batch_empty_input():
    emb = _get_embeddings_module()
    assert emb.embed_batch([]) == []


def test_serialize_deserialize_roundtrip():
    emb = _get_embeddings_module()
    vec = emb.embed("roundtrip test")
    assert vec is not None
    blob = emb.serialize_f32(vec)
    restored = emb.deserialize_f32(blob)
    # Floating point roundtrip should be within tolerance
    assert len(restored) == len(vec)
    for a, b in zip(vec, restored):
        assert abs(a - b) < 1e-6
