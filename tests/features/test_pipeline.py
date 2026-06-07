import pytest
from datetime import timedelta
from app.util.clock import now_utc
from app.features.pipeline import FeaturePipeline
from app.models.market import MarketData, FeatureSet

def make_market_data(n=30):
    bars = [{"t": now_utc() - timedelta(days=n-i),
             "o": 520.0+i*0.1, "h": 521.0+i*0.1, "l": 519.0+i*0.1,
             "c": 520.5+i*0.1, "v": 1_000_000+i*10_000}
            for i in range(n)]
    return MarketData(
        symbol="SPY", timestamp=now_utc(), pipeline_version="1.0.0",
        open=bars[-1]["o"], high=bars[-1]["h"], low=bars[-1]["l"],
        close=bars[-1]["c"], volume=bars[-1]["v"], bars_daily=bars,
        bid=521.0, ask=521.1, spread_proxy=0.0001, vix=18.5, news_sentiment=0.1,
    )

def test_compute_returns_feature_set():
    fs = FeaturePipeline.compute(make_market_data())
    assert isinstance(fs, FeatureSet)
    assert fs.symbol == "SPY"
    assert fs.feature_version == "1.0.0"

def test_all_numeric_fields_present():
    fs = FeaturePipeline.compute(make_market_data())
    for attr in ["price","ema_9","ema_20","ema_50","rsi","macd","atr","vix","relative_volume"]:
        assert isinstance(getattr(fs, attr), float), f"{attr} not float"

def test_relative_volume_positive():
    fs = FeaturePipeline.compute(make_market_data())
    assert fs.relative_volume > 0

def test_spread_proxy_passthrough():
    md = make_market_data()
    fs = FeaturePipeline.compute(md)
    assert fs.spread_proxy == md.spread_proxy
