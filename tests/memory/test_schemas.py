from app.memory.schemas import MemoryEpisode, RetrievedMemory, TradeReflection


def _episode(**kw):
    base = dict(kind="trade", symbol="SPY", summary="trend continuation win")
    base.update(kw)
    return MemoryEpisode(**base)


def test_episode_defaults():
    e = _episode()
    assert e.episode_id is not None
    assert e.outcome is None
    assert e.tags == []
    assert e.created_at is not None


def test_retrieved_memory_convenience_props():
    e = _episode(outcome="WIN", pnl_pct=1.2)
    rm = RetrievedMemory(episode=e, score=0.91, rank=0)
    assert rm.summary == "trend continuation win"
    assert rm.outcome == "WIN"
    assert rm.pnl_pct == 1.2


def test_trade_reflection_defaults():
    r = TradeReflection(trade_id="t1")
    assert r.strengths == []
    assert r.summary == ""
    assert r.reflection_id is not None
