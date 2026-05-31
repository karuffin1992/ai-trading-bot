import pytest
from uuid import uuid4
from datetime import datetime
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
    return e

def test_dry_run_returns_none():
    e = stub_engine(mode="dry_run")
    assert e.execute(make_signal(), account_balance=100.0) is None
    e._trading_client.submit_order.assert_not_called()

def test_paper_auto_submits(mocker):
    e = stub_engine()
    fill = FillRecord(execution_id=uuid4(), fill_price=521.05,
                      fill_time=datetime.utcnow(), slippage=0.05, broker_state="filled")
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
    result = e.execute(make_signal(), account_balance=100.0)
    assert result is not None
    assert result.broker_state == "stale"
