import math
from uuid import UUID
from datetime import datetime
from tests.golden import golden_runner as gr
from app.models.market import MarketData


def _uptrend_bars(n=60):
    bars = []
    price = 100.0
    for i in range(n):
        price += 0.25
        if i % 5 == 0:
            price -= 1.0
        o = price - 0.2
        h = price + 0.5
        lo = price - 0.6
        c = price
        v = 1_000_000 + (200_000 if i == n - 1 else 0)
        bars.append({"t": f"2026-0{1+i//28}-{(i%28)+1:02d}", "o": o, "h": h,
                     "l": lo, "c": c, "v": float(v)})
    return bars


def _market(vix=15.0, n=60):
    bars = _uptrend_bars(n)
    last = bars[-1]
    return MarketData(
        symbol="SPY", timestamp=datetime(2026, 6, 1), pipeline_version="1.0.0",
        open=last["o"], high=last["h"], low=last["l"], close=last["c"],
        volume=last["v"], spread_proxy=0.0001, vix=vix, news_sentiment=0.1,
        bars_daily=bars,
    )


def _case(md):
    return {"case_metadata": {"description": "t", "created_at": "2026-06-07T00:00:00",
                              "source": "synthetic", "expected_behavior": "x"},
            "market_data": md.model_dump(mode="json")}


def test_run_case_is_deterministic():
    case = _case(_market())
    r1 = gr.run_case(case)
    r2 = gr.run_case(case)
    assert r1["features_hash"] == r2["features_hash"]
    assert r1["signal_hash"] == r2["signal_hash"]
    assert r1["prompt_hash"] == r2["prompt_hash"]


def test_signal_case_produces_prompt_hash():
    r = gr.run_case(_case(_market(vix=15.0)))
    assert r["prompt_hash"] is not None
    assert r["prompt_preview"]
    assert r["retrieval_hash"] is None


def test_rejection_case_has_null_prompt_hash():
    r = gr.run_case(_case(_market(vix=99.0)))  # VIX over vix_max -> rejection
    assert r["prompt_hash"] is None
    assert r["prompt_preview"] is None


def test_trade_id_normalized_so_hash_is_stable():
    # Two independent runs would carry different random trade_ids if not normalized.
    r1 = gr.run_case(_case(_market()))
    r2 = gr.run_case(_case(_market()))
    assert r1["signal_hash"] == r2["signal_hash"]


def test_result_has_versions_and_timing():
    r = gr.run_case(_case(_market()))
    assert set(r["versions"]) == {"pipeline", "strategy", "prompt", "canonical_schema_version"}
    assert set(r["timing"]) == {"features_runtime_ms", "strategy_runtime_ms"}
