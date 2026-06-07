"""Regenerates the synthetic replay_cases/*.json fixtures.

Run: python -m tests.golden._gen_fixtures
Asserts each case exercises its intended path before writing, so a future data
tweak that silently changes behavior fails loudly here.
"""
import json
import math
import os
from datetime import datetime

from app.models.market import MarketData
from app.features.pipeline import FeaturePipeline
from app.strategies.spy_trend import SpyTrendStrategy
from app.models.signals import TradeSignal, TradeRejection

CASES_DIR = os.path.join(os.path.dirname(__file__), "replay_cases")


def _uptrend_bars(n=60):
    bars, price = [], 100.0
    for i in range(n):
        price += 0.25
        if i % 5 == 0:          # periodic dip keeps RSI out of overbought
            price -= 1.0
        bars.append({
            "t": f"2026-{1 + i // 28:02d}-{(i % 28) + 1:02d}",
            "o": round(price - 0.2, 4), "h": round(price + 0.5, 4),
            "l": round(price - 0.6, 4), "c": round(price, 4),
            "v": 1_000_000.0 + (300_000.0 if i == n - 1 else 0.0),
        })
    return bars


def _md(bars, vix):
    last = bars[-1]
    return MarketData(
        symbol="SPY", timestamp=datetime(2026, 6, 1), pipeline_version="1.0.0",
        open=last["o"], high=last["h"], low=last["l"], close=last["c"],
        volume=last["v"], spread_proxy=0.0001, vix=vix, news_sentiment=0.1,
        bars_daily=bars,
    )


def _envelope(md, description, expected_behavior):
    return {
        "case_metadata": {
            "description": description,
            "created_at": "2026-06-07T00:00:00",
            "source": "synthetic",
            "expected_behavior": expected_behavior,
        },
        "market_data": md.model_dump(mode="json"),
    }


def _write(name, env):
    os.makedirs(CASES_DIR, exist_ok=True)
    with open(os.path.join(CASES_DIR, name), "w", encoding="utf-8", newline="\n") as f:
        json.dump(env, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {name}")


def main():
    # 1. long_signal — full uptrend, all gates clear, expect a TradeSignal.
    md = _md(_uptrend_bars(60), vix=15.0)
    res = SpyTrendStrategy.evaluate(FeaturePipeline.compute(md))
    assert isinstance(res, TradeSignal), f"long_signal expected signal, got {res!r}"
    _write("long_signal.json",
           _envelope(md, "SPY uptrend, gates clear, long signal",
                     "TradeSignal generated; prompt built"))

    # 2. high_vix_rejection — same trend but VIX above vix_max -> rejection.
    md = _md(_uptrend_bars(60), vix=99.0)
    res = SpyTrendStrategy.evaluate(FeaturePipeline.compute(md))
    assert isinstance(res, TradeRejection), f"expected rejection, got {res!r}"
    assert any("VIX" in r for r in res.reasons)
    _write("high_vix_rejection.json",
           _envelope(md, "VIX above max -> rejection",
                     "TradeRejection; prompt_hash null"))

    # 3. insufficient_bars_nan — too few bars -> NaN indicators.
    md = _md(_uptrend_bars(5), vix=15.0)
    feats = FeaturePipeline.compute(md)
    assert math.isnan(feats.rsi), "expected NaN rsi with insufficient bars"
    _write("insufficient_bars_nan.json",
           _envelope(md, "insufficient bars -> NaN indicators",
                     "FeatureSet/signal carry NaN; verifies NaN canonicalization"))


if __name__ == "__main__":
    main()
