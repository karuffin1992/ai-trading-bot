import math

from app.memory.embeddings import DeterministicHashEmbedding, make_embedder


def test_deterministic_same_text_same_vector():
    e = DeterministicHashEmbedding(dim=64)
    assert e.embed("strong trend continuation") == e.embed("strong trend continuation")


def test_l2_normalized():
    e = DeterministicHashEmbedding(dim=64)
    v = e.embed("momentum breakout above vwap")
    assert abs(math.sqrt(sum(x * x for x in v)) - 1.0) < 1e-9


def test_dim_respected():
    e = DeterministicHashEmbedding(dim=128)
    assert len(e.embed("anything")) == 128


def test_empty_text_zero_vector():
    e = DeterministicHashEmbedding(dim=32)
    assert e.embed("") == [0.0] * 32


def test_different_text_differs():
    e = DeterministicHashEmbedding(dim=256)
    assert e.embed("bullish win") != e.embed("bearish loss")


def test_make_embedder_factory():
    e = make_embedder("deterministic_hash", dim=16, embedding_version="det-1.0.0")
    assert e.dim == 16
    assert e.embedding_version == "det-1.0.0"
