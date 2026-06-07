import pytest
from sqlalchemy import create_engine

from app.memory.embeddings import DeterministicHashEmbedding
from app.memory.episodic_store import EpisodicStore
from app.memory.reflection import ReflectionEngine, _classify
from app.persistence.db import Base, get_session


@pytest.fixture
def sf():
    e = create_engine("sqlite:///data/test_reflection.db")
    Base.metadata.create_all(e)
    yield lambda: get_session(e)
    Base.metadata.drop_all(e)


def _pos(pnl=12.0, **kw):
    base = dict(trade_id="t1", symbol="SPY", direction="long", qty=2.0,
                entry_price=500.0, exit_price=506.0, pnl=pnl)
    base.update(kw)
    return base


def test_classify():
    assert _classify(5.0) == "WIN"
    assert _classify(-5.0) == "LOSS"
    assert _classify(0.0) == "BREAKEVEN"
    assert _classify(None) == "BREAKEVEN"


def test_win_reflection_persisted(sf):
    eng = ReflectionEngine(sf, episodic_store=None, enabled=True)
    r = eng.reflect_on_position(_pos(pnl=12.0))
    assert r is not None
    assert r.outcome == "WIN"
    assert r.strengths and not r.weaknesses
    from app.persistence.db import MemoryReflectionRecord
    with sf() as s:
        rec = s.query(MemoryReflectionRecord).first()
        assert rec.outcome == "WIN"
        assert rec.trade_id == "t1"


def test_loss_reflection(sf):
    eng = ReflectionEngine(sf, enabled=True)
    r = eng.reflect_on_position(_pos(pnl=-8.0))
    assert r.outcome == "LOSS"
    assert r.weaknesses and not r.strengths


def test_reflection_writes_memory_episode(sf):
    store = EpisodicStore(sf, embedder=DeterministicHashEmbedding(dim=32))
    eng = ReflectionEngine(sf, episodic_store=store, enabled=True)
    eng.reflect_on_position(_pos())
    from app.persistence.db import MemoryEpisodeRecord
    with sf() as s:
        ep = s.query(MemoryEpisodeRecord).filter_by(kind="reflection").first()
        assert ep is not None
        assert ep.symbol == "SPY"
        assert ep.embedding_id is not None


def test_disabled_returns_none(sf):
    eng = ReflectionEngine(sf, enabled=False)
    assert eng.reflect_on_position(_pos()) is None


def test_no_session_factory_returns_none():
    eng = ReflectionEngine(None, enabled=True)
    assert eng.reflect_on_position(_pos()) is None


def test_reflect_on_closed_batch(sf):
    eng = ReflectionEngine(sf, enabled=True)
    out = eng.reflect_on_closed([_pos(trade_id="a"), _pos(trade_id="b", pnl=-3.0)])
    assert len(out) == 2
    assert {r.outcome for r in out} == {"WIN", "LOSS"}
