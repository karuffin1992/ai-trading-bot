import pytest
from uuid import uuid4
from app.util.clock import now_utc
from unittest.mock import MagicMock
from app.execution.executor import ExecutionEngine
from app.models.signals import TradeSignal
from app.models.execution import OrderResult, FillRecord

def make_signal():
    return TradeSignal(trade_id=uuid4(), symbol="SPY",
                       strategy="spy_trend_following", direction="long",
                       strategy_confidence=0.74, entry_price=521.0,
                       stop_loss=516.8, take_profit=527.3)

def stub_engine(mode="paper_auto"):
    e = ExecutionEngine.__new__(ExecutionEngine)
    e._trading_mode = mode
    mock_order = MagicMock(); mock_order.id = "order-123"
    e._trading_client = MagicMock()
    e._trading_client.submit_order.return_value = mock_order
    e._is_already_executed = lambda tid: False
    e._persist_execution = lambda *a, **kw: None
    e._positions = MagicMock()
    return e

def test_dry_run_returns_none():
    e = stub_engine(mode="dry_run")
    assert e.execute(make_signal(), account_balance=100.0) is None
    e._trading_client.submit_order.assert_not_called()

def test_paper_auto_submits(mocker):
    e = stub_engine()
    fill = FillRecord(execution_id=uuid4(), fill_price=521.05,
                      fill_time=now_utc(), slippage=0.05, broker_state="filled")
    mocker.patch.object(e, "_poll_fill", return_value=fill)
    result = e.execute(make_signal(), account_balance=100.0)
    e._trading_client.submit_order.assert_called_once()
    assert result is not None

def test_idempotency_blocks_duplicate(mocker):
    e = stub_engine()
    e._is_already_executed = lambda tid: True
    result = e.execute(make_signal(), account_balance=100.0)
    assert result is None
    e._trading_client.submit_order.assert_not_called()

def test_stale_fill_on_timeout(mocker):
    e = stub_engine()
    mocker.patch.object(e, "_poll_fill", return_value=None)
    mocker.patch.object(e, "_reconcile", return_value="stale")
    result = e.execute(make_signal(), account_balance=100.0)
    assert result is not None
    assert result.broker_state == "stale"

def test_reconcile_on_timeout(mocker):
    e = stub_engine()
    mocker.patch.object(e, "_poll_fill", return_value=None)
    mocker.patch.object(e, "_reconcile", return_value="reconciled")
    result = e.execute(make_signal(), account_balance=100.0)
    assert result.broker_state == "reconciled"
    e._positions.record_open.assert_not_called()

def test_records_open_on_fill(mocker):
    e = stub_engine()
    fill = FillRecord(execution_id=uuid4(), fill_price=521.05,
                      fill_time=now_utc(), slippage=0.05, broker_state="filled")
    mocker.patch.object(e, "_poll_fill", return_value=fill)
    e.execute(make_signal(), account_balance=100.0)
    e._positions.record_open.assert_called_once()

def test_bracket_fallback_to_market(mocker):
    e = stub_engine()
    e._trading_client.submit_order.side_effect = RuntimeError("bracket unsupported")
    fallback_order = MagicMock(); fallback_order.id = "fb-1"
    mocker.patch.object(e, "_submit_market_fallback", return_value=fallback_order)
    fill = FillRecord(execution_id=uuid4(), fill_price=521.0,
                      fill_time=now_utc(), slippage=0.0, broker_state="filled")
    mocker.patch.object(e, "_poll_fill", return_value=fill)
    result = e.execute(make_signal(), account_balance=100.0)
    assert result is not None and result.broker_order_id == "fb-1"

def test_bracket_fallback_gives_up(mocker):
    e = stub_engine()
    e._trading_client.submit_order.side_effect = RuntimeError("bracket unsupported")
    mocker.patch.object(e, "_submit_market_fallback", return_value=None)
    assert e.execute(make_signal(), account_balance=100.0) is None
