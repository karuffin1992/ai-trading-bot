from uuid import uuid4
from app.util.clock import now_utc
from app.models.market import MarketData, FeatureSet
from app.models.signals import TradeSignal, TradeRejection
from app.models.ai import AIAnalysis
from app.models.risk import RiskDecision, ValidationResult, KillSwitchState
from app.models.execution import OrderResult, FillRecord
from app.pipeline.context import PipelineContext

def test_feature_set_has_version():
    fs = FeatureSet(
        symbol="SPY", timeframe="1D", timestamp=now_utc(),
        feature_version="1.0.0", price=521.0, vwap=520.0,
        ema_9=521.5, ema_20=519.0, ema_50=515.0,
        rsi=55.0, macd=0.5, macd_signal=0.3, macd_hist=0.2,
        atr=2.1, vix=18.0, relative_volume=1.5, spread_proxy=0.0003,
        news_sentiment=0.2,
    )
    assert fs.feature_version == "1.0.0"

def test_trade_signal_type_literal():
    sig = TradeSignal(
        symbol="SPY", strategy="spy_trend_following",
        direction="long", strategy_confidence=0.74,
        entry_price=521.0, stop_loss=516.8, take_profit=527.3,
    )
    assert sig.type == "SIGNAL"

def test_trade_rejection_has_reasons():
    rej = TradeRejection(
        symbol="SPY", strategy="spy_trend_following",
        reasons=["VIX too high"], strategy_confidence=0.3,
    )
    assert rej.type == "REJECTION"
    assert len(rej.reasons) == 1

def test_ai_analysis_separates_confidence():
    a = AIAnalysis(
        decision="APPROVE", ai_confidence=0.81, regime="bullish",
        reasoning="ok", risk_factors=[], no_trade_reasons=[],
        raw_prompt="p", raw_response="r",
        model_version="claude-sonnet-4-6", prompt_version="1.0.0",
    )
    assert a.ai_confidence == 0.81
    assert not hasattr(a, "strategy_confidence")

def test_pipeline_context_starts_empty():
    ctx = PipelineContext()
    assert ctx.market_data is None
    assert ctx.signal is None
    assert ctx.errors == []
