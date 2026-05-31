import json
from unittest.mock import MagicMock
from datetime import datetime
from uuid import uuid4
from app.ai.analyst import AIAnalyst
from app.models.signals import TradeSignal, TradeRejection
from app.models.ai import AIAnalysis
from app.models.market import FeatureSet

def make_fs():
    return FeatureSet(symbol="SPY", timeframe="1D", timestamp=datetime.utcnow(),
                      feature_version="1.0.0", price=521.0, vwap=519.0,
                      ema_9=521.5, ema_20=519.0, ema_50=515.0, rsi=55.0,
                      macd=0.5, macd_signal=0.3, macd_hist=0.2, atr=2.1,
                      vix=18.0, relative_volume=1.5, spread_proxy=0.0003,
                      news_sentiment=0.1)

def make_signal():
    return TradeSignal(trade_id=uuid4(), symbol="SPY",
                       strategy="spy_trend_following", direction="long",
                       strategy_confidence=0.74, entry_price=521.0,
                       stop_loss=516.8, take_profit=527.3)

def mock_analyst(response_dict: dict) -> AIAnalyst:
    a = AIAnalyst.__new__(AIAnalyst)
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps(response_dict))]
    client = MagicMock()
    client.messages.create.return_value = msg
    a._client = client
    return a

def test_rejection_skips_llm():
    a = AIAnalyst.__new__(AIAnalyst)
    a._client = MagicMock()
    rej = TradeRejection(symbol="SPY", strategy="spy_trend_following",
                         reasons=["VIX too high"], strategy_confidence=0.3)
    result = a.analyze(rej, make_fs())
    a._client.messages.create.assert_not_called()
    assert result.decision == "REJECT"

def test_approve_parses_correctly():
    a = mock_analyst({"decision":"APPROVE","ai_confidence":0.82,"regime":"bullish",
                      "reasoning":"ok","risk_factors":[],"no_trade_reasons":[]})
    r = a.analyze(make_signal(), make_fs())
    assert isinstance(r, AIAnalysis)
    assert r.decision == "APPROVE"
    assert r.ai_confidence == 0.82
    assert r.failed is False

def test_parse_failure_returns_failed_analysis():
    a = AIAnalyst.__new__(AIAnalyst)
    msg = MagicMock()
    msg.content = [MagicMock(text="not json {{")]
    a._client = MagicMock()
    a._client.messages.create.return_value = msg
    r = a.analyze(make_signal(), make_fs())
    assert r.failed is True
    assert r.decision == "NO_TRADE"

def test_raw_fields_stored():
    a = mock_analyst({"decision":"APPROVE","ai_confidence":0.75,"regime":"neutral",
                      "reasoning":"ok","risk_factors":[],"no_trade_reasons":[]})
    r = a.analyze(make_signal(), make_fs())
    assert len(r.raw_prompt) > 0
    assert len(r.raw_response) > 0
