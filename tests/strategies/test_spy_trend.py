import pytest
from app.util.clock import now_utc
from app.strategies.spy_trend import SpyTrendStrategy
from app.models.market import FeatureSet
from app.models.signals import TradeSignal, TradeRejection

def make_fs(**ov):
    d = dict(symbol="SPY", timeframe="1D", timestamp=now_utc(),
             feature_version="1.0.0", price=521.0, vwap=519.0,
             ema_9=521.5, ema_20=519.0, ema_50=515.0,
             rsi=55.0, macd=0.5, macd_signal=0.3, macd_hist=0.2,
             atr=2.1, vix=18.0, relative_volume=1.5, spread_proxy=0.0003,
             news_sentiment=0.1)
    d.update(ov)
    return FeatureSet(**d)

def test_all_pass_returns_signal():
    assert isinstance(SpyTrendStrategy.evaluate(make_fs()), TradeSignal)

def test_high_vix_returns_rejection():
    r = SpyTrendStrategy.evaluate(make_fs(vix=30.0))
    assert isinstance(r, TradeRejection)
    assert any("VIX" in x for x in r.reasons)

def test_rsi_overbought_returns_rejection():
    r = SpyTrendStrategy.evaluate(make_fs(rsi=75.0))
    assert isinstance(r, TradeRejection)
    assert any("RSI" in x for x in r.reasons)

def test_price_below_ema20_returns_rejection():
    assert isinstance(SpyTrendStrategy.evaluate(make_fs(price=515.0, ema_20=520.0)), TradeRejection)

def test_low_volume_returns_rejection():
    assert isinstance(SpyTrendStrategy.evaluate(make_fs(relative_volume=0.8)), TradeRejection)

def test_stop_below_entry():
    r = SpyTrendStrategy.evaluate(make_fs())
    assert isinstance(r, TradeSignal)
    assert r.stop_loss < r.entry_price

def test_tp_above_entry():
    r = SpyTrendStrategy.evaluate(make_fs())
    assert isinstance(r, TradeSignal)
    assert r.take_profit > r.entry_price

def test_confidence_monotonic():
    strong = SpyTrendStrategy.evaluate(make_fs(relative_volume=2.5, ema_9=525.0, ema_20=515.0))
    weak   = SpyTrendStrategy.evaluate(make_fs(relative_volume=1.3, ema_9=521.0, ema_20=520.5))
    assert isinstance(strong, TradeSignal)
    assert isinstance(weak, TradeSignal)
    assert strong.strategy_confidence >= weak.strategy_confidence
