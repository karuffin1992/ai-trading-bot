import pytest
from app.persistence.db import (Base, PositionRecord, DailySummaryRecord,
                                make_engine, get_session, make_session_factory)
from app.execution.position_manager import PositionManager
from app.util.clock import et_today_iso

@pytest.fixture
def sf(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/t.db")
    Base.metadata.create_all(engine)
    return make_session_factory(engine)

def _open_position(sf, symbol="SPY", entry=500.0, qty=2.0):
    pm = PositionManager(sf)
    pm.record_open("trade-1", symbol, "long", qty, entry, 495.0, 510.0)
    return pm

def test_record_open_writes_position(sf):
    _open_position(sf)
    with sf() as s:
        rows = s.query(PositionRecord).filter_by(status="open").all()
        assert len(rows) == 1 and rows[0].symbol == "SPY"

def test_sync_closes_when_broker_flat(sf):
    pm = _open_position(sf, entry=500.0, qty=2.0)
    pm._broker_open_map = lambda: {}          # broker shows no open position
    pm._last_price = lambda sym: 506.0        # exit price
    pm.sync()
    with sf() as s:
        r = s.query(PositionRecord).first()
        assert r.status == "closed"
        assert r.pnl == 12.0                   # (506-500)*2
        assert s.get(DailySummaryRecord, et_today_iso()) is not None

def test_sync_keeps_open_when_broker_open(sf):
    pm = _open_position(sf)
    pm._broker_open_map = lambda: {"SPY": {"unrealized_pl": 4.0, "current_price": 502.0}}
    pm.sync()
    with sf() as s:
        r = s.query(PositionRecord).first()
        assert r.status == "open" and r.pnl == 4.0

def test_close_all_flattens(sf):
    pm = _open_position(sf, entry=500.0, qty=1.0)
    pm._flatten = lambda sym: 497.0
    pm.close_all()
    with sf() as s:
        r = s.query(PositionRecord).first()
        assert r.status == "closed" and r.pnl == -3.0
