import pytest
from sqlalchemy import create_engine

from app.memory.embeddings import DeterministicHashEmbedding
from app.memory.episodic_store import EpisodicStore
from app.memory.schemas import MemoryEpisode
from app.persistence.db import Base, get_session


@pytest.fixture
def sf():
    e = create_engine("sqlite:///data/test_memory_store.db")
    Base.metadata.create_all(e)
    yield lambda: get_session(e)
    Base.metadata.drop_all(e)


def _episode(summary="trend continuation win", **kw):
    base = dict(kind="trade", symbol="SPY", summary=summary)
    base.update(kw)
    return MemoryEpisode(**base)


def test_write_and_get_roundtrip(sf):
    store = EpisodicStore(sf, embedder=DeterministicHashEmbedding(dim=32))
    ep = _episode(outcome="WIN", pnl_pct=1.2)
    eid = store.write_episode(ep)
    got = store.get_episode(eid)
    assert got is not None
    assert got.summary == "trend continuation win"
    assert got.outcome == "WIN"
    assert got.embedding_id is not None


def test_embedding_row_created(sf):
    store = EpisodicStore(sf, embedder=DeterministicHashEmbedding(dim=32))
    eid = store.write_episode(_episode())
    from app.persistence.db import EmbeddingRegistryRecord, MemoryEpisodeRecord
    with sf() as s:
        rec = s.get(MemoryEpisodeRecord, eid)
        emb = s.get(EmbeddingRegistryRecord, rec.embedding_id)
        assert emb is not None
        assert emb.dim == 32
        assert len(emb.vector_json) == 32
        assert emb.provider == "deterministic_hash"


def test_recent_filters_by_symbol_and_kind(sf):
    store = EpisodicStore(sf, embedder=DeterministicHashEmbedding(dim=16))
    store.write_episode(_episode(summary="a", symbol="SPY", kind="trade"))
    store.write_episode(_episode(summary="b", symbol="QQQ", kind="trade"))
    store.write_episode(_episode(summary="c", symbol="SPY", kind="reflection"))
    spy_trades = store.recent("SPY", kind="trade")
    assert len(spy_trades) == 1
    assert spy_trades[0].summary == "a"


def test_no_session_factory_is_noop():
    store = EpisodicStore(None, embedder=DeterministicHashEmbedding(dim=8))
    assert store.write_episode(_episode()) is None
    assert store.get_episode("x") is None
    assert store.recent("SPY") == []
