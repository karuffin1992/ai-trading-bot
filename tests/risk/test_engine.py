import pytest
from uuid import uuid4
from datetime import datetime
from app.risk.engine import RiskEngine
from app.models.signals import TradeSignal
from app.models.ai import AIAnalysis
from app.models.risk import RiskDecision, KillSwitchState

def make_signal(conf=0.74):
    return TradeSignal(trade_id=uuid4(), symbol="SPY",
                       strategy="spy_trend_following", direction="long",
                       strategy_confidence=conf, entry_price=521.0,
                       stop_loss=516.8, take_profit=527.3)

def make_analysis(conf=0.82, decision="APPROVE"):
    return AIAnalysis(decision=decision, ai_confidence=conf, regime="bullish",
                      reasoning="ok", risk_factors=[], no_trade_reasons=[],
                      raw_prompt="", raw_response="",
                      model_version="claude-sonnet-4-6", prompt_version="1.0.0")

def stub_engine(balance=100.0, daily_trades=0, drawdown=0.0,
                consec=0, kill=False, last_trade=None):
    e = RiskEngine.__new__(RiskEngine)
    e._get_account_balance    = lambda: balance
    e._get_daily_trade_count  = lambda: daily_trades
    e._get_daily_drawdown     = lambda: drawdown
    e._get_consecutive_losses = lambda: consec
    e._get_kill_switch        = lambda: KillSwitchState(active=kill)
    e._get_last_trade_time    = lambda: last_trade
    e._is_earnings_proximity  = lambda sym: False
    e._activate_kill_switch   = lambda r: None
    return e

def test_approved_when_clear():
    assert stub_engine().evaluate(make_signal(), make_analysis(), vix=18.0).outcome == "APPROVED"

def test_kill_switch_active():
    r = stub_engine(kill=True).evaluate(make_signal(), make_analysis(), vix=18.0)
    assert r.outcome == "KILL" and r.tier == "kill_switch"

def test_drawdown_triggers_kill():
    assert stub_engine(drawdown=0.12).evaluate(make_signal(), make_analysis(), vix=18.0).outcome == "KILL"

def test_consec_losses_triggers_kill():
    assert stub_engine(consec=3).evaluate(make_signal(), make_analysis(), vix=18.0).outcome == "KILL"

def test_daily_limit_blocks():
    r = stub_engine(daily_trades=1).evaluate(make_signal(), make_analysis(), vix=18.0)
    assert r.outcome == "BLOCKED" and r.tier == "hard_block"

def test_low_balance_blocks():
    assert stub_engine(balance=5.0).evaluate(make_signal(), make_analysis(), vix=18.0).outcome == "BLOCKED"

def test_low_risk_score_blocks():
    r = stub_engine().evaluate(make_signal(conf=0.3), make_analysis(conf=0.3), vix=24.0)
    assert r.outcome == "BLOCKED" and r.tier == "score"

def test_risk_score_in_result():
    r = stub_engine().evaluate(make_signal(), make_analysis(), vix=18.0)
    assert 0.0 <= r.risk_score <= 1.0
