from datetime import datetime

from app.memory.schemas import MemoryEpisode, RetrievedMemory, TradeReflection
from app.memory.summarizer import Summarizer, _truncate


def _rm(summary, outcome="WIN", pnl_pct=1.2, rank=0):
    ep = MemoryEpisode(kind="trade", symbol="SPY", summary=summary,
                       outcome=outcome, pnl_pct=pnl_pct,
                       created_at=datetime(2026, 5, 10, 14, 0, 0))
    return RetrievedMemory(episode=ep, score=0.9, rank=rank)


def test_truncate():
    assert _truncate("abc", 10) == "abc"
    out = _truncate("abcdefghij", 5)
    assert len(out) <= 5
    assert out.endswith("…")


def test_summarize_episodes_formats_lines():
    block = Summarizer().summarize_episodes([_rm("trend win", rank=0),
                                             _rm("momentum loss", "LOSS", -0.8, 1)],
                                            token_budget=200)
    lines = block.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("1. 2026-05-10 | WIN (+1.2%)")
    assert "momentum loss" in lines[1]


def test_summarize_episodes_respects_budget():
    episodes = [_rm("x" * 100, rank=i) for i in range(10)]
    block = Summarizer().summarize_episodes(episodes, token_budget=30)  # ~120 chars
    # Budget is small, so not all 10 lines fit.
    assert len(block.splitlines()) < 10


def test_summarize_reflection():
    r = TradeReflection(trade_id="t1", summary="WIN on SPY")
    assert Summarizer().summarize_reflection(r) == "WIN on SPY"
