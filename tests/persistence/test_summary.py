import pytest
from app.persistence.db import (Base, PositionRecord, DailySummaryRecord,
                                make_engine, get_session)
from app.persistence.summary import compute_daily_summary, _peak_to_trough_drawdown
from app.util.clock import now_utc, et_today_iso

@pytest.fixture
def session(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/t.db")
    Base.metadata.create_all(engine)
    with get_session(engine) as s:
        yield s

def _pos(s, pnl, status="closed"):
    s.add(PositionRecord(id=f"p{pnl}{status}", trade_id="t", symbol="SPY",
                         direction="long", qty=1, entry_price=500, stop_loss=495,
                         take_profit=510, status=status, pnl=pnl,
                         opened_at=now_utc(), closed_at=now_utc()))
    s.commit()

def test_summary_aggregates(session):
    _pos(session, 5.0); _pos(session, -2.0); _pos(session, 3.0)
    rec = compute_daily_summary(session, starting_capital=100.0)
    assert rec.trade_count == 3
    assert rec.total_pnl == 6.0
    assert rec.win_rate == round(2/3, 4)
    assert rec.profit_factor == round(8.0/2.0, 4)

def test_summary_consecutive_losses_trailing(session):
    _pos(session, 5.0); _pos(session, -1.0); _pos(session, -2.0)
    rec = compute_daily_summary(session, starting_capital=100.0)
    assert rec.consecutive_losses == 2

def test_summary_ignores_open_positions(session):
    _pos(session, 5.0, status="open")
    rec = compute_daily_summary(session, starting_capital=100.0)
    assert rec.trade_count == 0

def test_summary_persisted_by_date(session):
    _pos(session, 1.0)
    compute_daily_summary(session, starting_capital=100.0)
    assert session.get(DailySummaryRecord, et_today_iso()) is not None

def test_drawdown_peak_to_trough():
    # equity: 100 -> 110 -> 104 -> 108 ; peak 110, trough 104 -> dd 6/110
    assert _peak_to_trough_drawdown([10.0, -6.0, 4.0], 100.0) == round(6/110, 4)
