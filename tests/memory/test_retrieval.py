import pytest
from sqlalchemy import create_engine

from app.memory.embeddings import DeterministicHashEmbedding
from app.memory.episodic_store import EpisodicStore
from app.memory.retrieval import MemoryRetriever, SQLiteVectorStore, _cosine
from app.memory.schemas import MemoryEpisode
from app.persistence.db import Base, get_session


@pytest.fixture
def sf():
    e = create_engine("sqlite:///data/test_memory_retrieval.db")
    Base.metadata.create_all(e)
    yield lambda: get_session(e)
    Base.metadata.drop_all(e)


def _seed(sf, embedder):
    store = EpisodicStore(sf, embedder=embedder)
    store.write_episode(MemoryEpisode(kind="trade", symbol="SPY",
                                      summary="strong bullish trend continuation win",
                                      outcome="WIN", pnl_pct=1.2))
    store.write_episode(MemoryEpisode(kind="trade", symbol="SPY",
                                      summary="weak momentum confirmation loss",
                                      outcome="LOSS", pnl_pct=-0.8))
    store.write_episode(MemoryEpisode(kind="trade", symbol="QQQ",
                                      summary="bullish trend continuation win",
                                      outcome="WIN", pnl_pct=0.9))


def test_cosine_basic():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert _cosine([], [1.0]) == 0.0


def test_retrieve_ranks_similar_first(sf):
    embedder = DeterministicHashEmbedding(dim=256)
    _seed(sf, embedder)
    store = SQLiteVectorStore(sf)
    r = MemoryRetriever(embedder, store, session_factory=sf)
    hits = r.retrieve("bullish trend continuation", k=3)
    assert len(hits) == 3
    assert hits[0].rank == 0
    # Most similar summary should outrank the "weak momentum loss" one.
    assert "trend continuation" in hits[0].summary
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_filter_by_symbol(sf):
    embedder = DeterministicHashEmbedding(dim=256)
    _seed(sf, embedder)
    store = SQLiteVectorStore(sf)
    r = MemoryRetriever(embedder, store, session_factory=sf)
    hits = r.retrieve("trend continuation", k=5, filters={"symbol": "QQQ"})
    assert len(hits) == 1
    assert hits[0].episode.symbol == "QQQ"


def test_retrieve_no_session_factory_empty():
    embedder = DeterministicHashEmbedding(dim=16)
    r = MemoryRetriever(embedder, SQLiteVectorStore(None), session_factory=None)
    assert r.retrieve("anything") == []
