import json
from tests.golden import golden_runner as gr


def test_explain_text_contains_stage_hashes(capsys):
    case = {"case_metadata": {"description": "d", "created_at": "2026-06-07T00:00:00",
                              "source": "synthetic", "expected_behavior": "x"},
            "market_data": _signal_market().model_dump(mode="json")}
    result = gr.run_case(case)
    gr.print_explain(case, result, expected=None)
    out = capsys.readouterr().out
    assert "features_hash" in out
    assert result["features_hash"] in out
    assert "FEATURES" in out  # stage header


def test_explain_diff_flags_changed_field(capsys):
    case = {"case_metadata": {"description": "d", "created_at": "2026-06-07T00:00:00",
                              "source": "synthetic", "expected_behavior": "x"},
            "market_data": _signal_market().model_dump(mode="json")}
    result = gr.run_case(case)
    stale = gr.expected_payload(result)
    stale["features_hash"] = "0" * 64  # pretend the blessed hash is stale
    gr.print_explain(case, result, expected=stale)
    out = capsys.readouterr().out
    assert "MISMATCH" in out and "features_hash" in out


def _signal_market():
    from datetime import datetime
    from app.models.market import MarketData
    bars = []
    price = 100.0
    for i in range(60):
        price += 0.25
        if i % 5 == 0:
            price -= 1.0
        bars.append({"t": f"2026-06-{(i % 28) + 1:02d}", "o": price - 0.2,
                     "h": price + 0.5, "l": price - 0.6, "c": price,
                     "v": 1_000_000.0 + (200_000.0 if i == 59 else 0.0)})
    last = bars[-1]
    return MarketData(symbol="SPY", timestamp=datetime(2026, 6, 1),
                      pipeline_version="1.0.0", open=last["o"], high=last["h"],
                      low=last["l"], close=last["c"], volume=last["v"],
                      spread_proxy=0.0001, vix=15.0, news_sentiment=0.1, bars_daily=bars)
