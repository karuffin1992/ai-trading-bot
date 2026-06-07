from app.ai.analyst import AIAnalyst, _load
from app.models.market import FeatureSet
from app.models.signals import TradeSignal
from app.config import settings
from datetime import datetime


def _features():
    return FeatureSet(
        symbol="SPY", timeframe="1D", timestamp=datetime(2026, 6, 1),
        feature_version="1.0.0", price=500.0, vwap=499.0,
        ema_9=499.5, ema_20=498.0, ema_50=495.0, rsi=55.0,
        macd=1.0, macd_signal=0.8, macd_hist=0.2, atr=3.0,
        vix=15.0, relative_volume=1.5, spread_proxy=0.0001, news_sentiment=0.1,
    )


def _signal():
    return TradeSignal(
        symbol="SPY", strategy="spy_trend_following", direction="long",
        strategy_confidence=0.7, entry_price=500.0,
        stop_loss=494.0, take_profit=509.0, position_side="long",
    )


def test_build_prompt_matches_legacy_format():
    a = AIAnalyst.__new__(AIAnalyst)  # no client
    f, s = _features(), _signal()
    ctx = a.build_prompt_context(s, f, account_balance=100.0)
    prompt = a.build_prompt(ctx)

    expected = _load("market_analysis.txt").format(
        trading_mode=settings.trading_mode,
        account_balance=100.0,
        feature_set_json=f.model_dump_json(indent=2),
        trade_signal_json=s.model_dump_json(indent=2),
    )
    assert prompt == expected  # flags off -> no memory header, byte-identical


def test_build_prompt_context_keys():
    a = AIAnalyst.__new__(AIAnalyst)
    ctx = a.build_prompt_context(_signal(), _features())
    assert set(ctx) == {
        "trading_mode", "account_balance",
        "feature_set_json", "trade_signal_json", "memory_block",
    }
    assert ctx["memory_block"] == ""  # retriever absent / injection disabled
