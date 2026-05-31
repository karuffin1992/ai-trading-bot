import pytest
from uuid import uuid4
from datetime import datetime
from app.validation.validator import TradeValidator
from app.models.signals import TradeSignal
from app.models.risk import ValidationResult

def make_signal():
    return TradeSignal(trade_id=uuid4(), symbol="SPY",
                       strategy="spy_trend_following", direction="long",
                       strategy_confidence=0.74, entry_price=521.0,
                       stop_loss=516.8, take_profit=527.3)

def stub(mode="paper_auto", spread=0.0002, power=100.0, open_pos=False):
    v = TradeValidator.__new__(TradeValidator)
    v._trading_mode    = mode
    v._get_spread      = lambda sym: spread
    v._get_buying_power= lambda: power
    v._has_open_position= lambda sym: open_pos
    v._write_pending   = lambda sig: None
    return v

def test_pass_when_clear():
    assert stub().validate(make_signal()).outcome == "PASS"

def test_fail_dry_run():
    r = stub(mode="dry_run").validate(make_signal())
    assert r.outcome == "FAIL" and "dry_run" in r.reason

def test_fail_wide_spread():
    r = stub(spread=0.01).validate(make_signal())
    assert r.outcome == "FAIL" and "spread" in r.reason.lower()

def test_fail_low_buying_power():
    assert stub(power=1.0).validate(make_signal()).outcome == "FAIL"

def test_fail_open_position():
    assert stub(open_pos=True).validate(make_signal()).outcome == "FAIL"

def test_live_manual_writes_pending():
    wrote = []
    v = stub(mode="live_manual")
    v._write_pending = lambda sig: wrote.append(sig)
    r = v.validate(make_signal())
    assert r.outcome == "FAIL" and "approval" in r.reason.lower()
    assert len(wrote) == 1
