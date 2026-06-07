# Golden Replay / Determinism Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic regression harness that replays frozen `MarketData` through features → signal → AI prompt and asserts per-stage SHA256 hashes stay stable.

**Architecture:** A canonicalizer rounds floats to 6dp and emits stable JSON; a runner replays each committed case and produces `features_hash`/`signal_hash`/`prompt_hash`; pytest parametrizes over cases and compares against blessed expected files. Prompt building is extracted from `AIAnalyst.analyze()` into two pure methods so prompts can be hashed with no LLM client and no network.

**Tech Stack:** Python 3.12, Pydantic v2, pandas / pandas_ta_classic, pytest. Stdlib `hashlib`, `json`, `math`, `textwrap`, `argparse`.

**Spec:** `docs/superpowers/specs/2026-06-07-golden-replay-tests-design.md`

---

## File structure

- Create `tests/golden/__init__.py` — package marker.
- Create `tests/golden/canonical.py` — `canonicalize(obj)` + `sha256_of(obj)`. Pure, no project imports.
- Create `tests/golden/golden_runner.py` — `run_case`, `load_case`, `load_expected`, `write_expected`, CLI (`--update`, `--explain`).
- Create `tests/golden/_gen_fixtures.py` — committed synthetic-fixture generator (documents how `replay_cases/*.json` were produced).
- Create `tests/golden/replay_cases/*.json` — input fixtures (generated, committed).
- Create `tests/golden/expected_hashes/*.json` — blessed outputs (generated via `--update`, committed).
- Create `tests/golden/test_canonical.py` — unit tests for the canonicalizer.
- Create `tests/golden/test_golden.py` — parametrized golden assertions.
- Modify `app/ai/analyst.py` — extract `build_prompt_context` + `build_prompt`; `analyze()` calls them.

---

## Task 1: Canonicalizer

**Files:**
- Create: `tests/golden/__init__.py`
- Create: `tests/golden/canonical.py`
- Test: `tests/golden/test_canonical.py`

- [ ] **Step 1: Create package marker**

Create `tests/golden/__init__.py` (empty file).

- [ ] **Step 2: Write the failing tests**

Create `tests/golden/test_canonical.py`:

```python
import json
import math
from decimal import Decimal
import numpy as np
from tests.golden.canonical import canonicalize, sha256_of


def test_float_rounded_to_6dp():
    assert canonicalize(1.234567891) == json.dumps(1.234568)
    # Sub-6dp noise collapses to the same canonical form.
    assert canonicalize(1.2345670001) == canonicalize(1.2345670002)


def test_nan_and_inf_become_strings():
    assert canonicalize(float("nan")) == json.dumps("NaN")
    assert canonicalize(float("inf")) == json.dumps("Infinity")
    assert canonicalize(float("-inf")) == json.dumps("-Infinity")


def test_decimal_coerced_to_rounded_float():
    assert canonicalize(Decimal("1.2345678")) == json.dumps(1.234568)


def test_numpy_scalars_coerced():
    assert canonicalize(np.float64(1.2345678)) == json.dumps(1.234568)
    assert canonicalize(np.int64(7)) == json.dumps(7)
    assert canonicalize(np.bool_(True)) == json.dumps(True)


def test_dict_keys_sorted_recursively():
    a = canonicalize({"b": 1, "a": {"d": 2, "c": 3}})
    b = canonicalize({"a": {"c": 3, "d": 2}, "b": 1})
    assert a == b
    assert a == '{"a":{"c":3,"d":2},"b":1}'


def test_list_order_preserved():
    assert canonicalize([3, 1, 2]) != canonicalize([1, 2, 3])
    assert canonicalize([3, 1, 2]) == "[3,1,2]"


def test_newlines_normalized():
    assert canonicalize("a\r\nb\rc") == canonicalize("a\nb\nc")


def test_sha256_stable_and_hex():
    h = sha256_of({"x": 1.0000001})
    assert h == sha256_of({"x": 1.00000009})  # same after 6dp round
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/golden/test_canonical.py -v`
Expected: FAIL — `ModuleNotFoundError: tests.golden.canonical`.

- [ ] **Step 4: Implement the canonicalizer**

Create `tests/golden/canonical.py`:

```python
import hashlib
import json
import math
from datetime import datetime, date
from decimal import Decimal

FLOAT_DP = 6


def _normalize(obj):
    # numpy scalars -> python primitives (no hard numpy dependency).
    if type(obj).__module__ == "numpy":
        item = getattr(obj, "item", None)
        if callable(item):
            obj = item()

    if isinstance(obj, bool):
        return obj
    if isinstance(obj, Decimal):
        obj = float(obj)
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        return round(obj, FLOAT_DP)
    if isinstance(obj, int):
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, str):
        return obj.replace("\r\n", "\n").replace("\r", "\n")
    if isinstance(obj, dict):
        # Keys coerced to str; values normalized; ordering handled at dump time.
        return {str(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]  # order preserved
    if obj is None:
        return None
    raise TypeError(f"canonicalize: unsupported type {type(obj)!r}")


def canonicalize(obj) -> str:
    return json.dumps(
        _normalize(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def sha256_of(obj) -> str:
    return hashlib.sha256(canonicalize(obj).encode("utf-8")).hexdigest()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/golden/test_canonical.py -v`
Expected: PASS (8 tests).

- [ ] **Step 6: Commit**

```bash
git add tests/golden/__init__.py tests/golden/canonical.py tests/golden/test_canonical.py
git commit -m "feat: golden test canonicalizer (rounded-float stable JSON hashing)"
```

---

## Task 2: Extract prompt building from AIAnalyst

**Files:**
- Modify: `app/ai/analyst.py:27-73` (`analyze`) and add two methods
- Test: `tests/ai/test_analyst_prompt.py`

The runner must build the prompt with no Anthropic client and no network. Split prompt
construction into `build_prompt_context` (gather values, pure) and `build_prompt` (format the
template). `analyze()` calls both. With memory flags off the output is byte-identical to today.

- [ ] **Step 1: Write the failing test**

Create `tests/ai/test_analyst_prompt.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ai/test_analyst_prompt.py -v`
Expected: FAIL — `AttributeError: 'AIAnalyst' object has no attribute 'build_prompt_context'`.

- [ ] **Step 3: Refactor analyst.py**

In `app/ai/analyst.py`, replace the prompt-building block inside `analyze()` (the
`system = _load(...)` line through the `if memory_block:` block, lines 38–51) with:

```python
        system = _load("system_prompt.txt")
        ctx = self.build_prompt_context(signal, features, account_balance)
        user = self.build_prompt(ctx)
```

Then add these two methods directly after `analyze()` (before `_call_model`):

```python
    # Pure context assembly for the prompt — no client, no network. Used by
    # analyze() and by the golden replay runner.
    def build_prompt_context(self, signal: TradeSignal, features: FeatureSet,
                             account_balance: float = 100.0) -> dict:
        return {
            "trading_mode": settings.trading_mode,
            "account_balance": account_balance,
            "feature_set_json": features.model_dump_json(indent=2),
            "trade_signal_json": signal.model_dump_json(indent=2),
            "memory_block": self._memory_block(signal, features),
        }

    # Formats the template from a context dict. Memory block concatenated under
    # its header only when present, so the disabled path is byte-identical to legacy.
    def build_prompt(self, ctx: dict) -> str:
        user = _load("market_analysis.txt").format(
            trading_mode=ctx["trading_mode"],
            account_balance=ctx["account_balance"],
            feature_set_json=ctx["feature_set_json"],
            trade_signal_json=ctx["trade_signal_json"],
        )
        if ctx.get("memory_block"):
            user = f"{user}\n\nRELEVANT PAST EPISODES:\n{ctx['memory_block']}"
        return user
```

- [ ] **Step 4: Run the new test plus existing analyst tests**

Run: `pytest tests/ai/test_analyst_prompt.py tests/ai/test_analyst.py tests/ai/test_analyst_memory.py -v`
Expected: PASS (new tests green; existing `test_analyst.py` / `test_analyst_memory.py` unchanged-green — proves zero prompt regression).

- [ ] **Step 5: Commit**

```bash
git add app/ai/analyst.py tests/ai/test_analyst_prompt.py
git commit -m "refactor: extract build_prompt_context/build_prompt from AIAnalyst.analyze"
```

---

## Task 3: Runner core (run_case + load/write helpers)

**Files:**
- Create: `tests/golden/golden_runner.py`
- Test: `tests/golden/test_runner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/golden/test_runner.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/golden/test_runner.py -v`
Expected: FAIL — `AttributeError: module 'tests.golden.golden_runner' has no attribute 'run_case'`.

- [ ] **Step 3: Implement runner core**

Create `tests/golden/golden_runner.py`:

```python
import json
import os
import time
import textwrap
from uuid import UUID

from app.models.market import MarketData
from app.models.signals import TradeSignal
from app.features.pipeline import FeaturePipeline
from app.strategies.spy_trend import SpyTrendStrategy
from app.ai.analyst import AIAnalyst
from app.config import settings
from tests.golden.canonical import canonicalize, sha256_of

CASES_DIR = os.path.join(os.path.dirname(__file__), "replay_cases")
EXPECTED_DIR = os.path.join(os.path.dirname(__file__), "expected_hashes")
CANONICAL_SCHEMA_VERSION = "1.0.0"
SENTINEL_TRADE_ID = UUID(int=0)
PREVIEW_WIDTH = 300


def _versions() -> dict:
    return {
        "pipeline": settings.pipeline_version,
        "strategy": settings.strategy_version,
        "prompt": settings.prompt_version,
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
    }


def run_case(case: dict) -> dict:
    md = MarketData(**case["market_data"])

    t0 = time.perf_counter()
    features = FeaturePipeline.compute(md)
    features_ms = (time.perf_counter() - t0) * 1000.0
    features_dump = features.model_dump(mode="json")
    features_hash = sha256_of(features_dump)

    t1 = time.perf_counter()
    result = SpyTrendStrategy.evaluate(features)
    strategy_ms = (time.perf_counter() - t1) * 1000.0

    if isinstance(result, TradeSignal):
        result = result.model_copy(update={"trade_id": SENTINEL_TRADE_ID})
    signal_dump = result.model_dump(mode="json")
    signal_hash = sha256_of(signal_dump)

    prompt_hash = None
    prompt_preview = None
    prompt = None
    if isinstance(result, TradeSignal):
        analyst = AIAnalyst.__new__(AIAnalyst)  # no client, no network
        ctx = analyst.build_prompt_context(result, features, account_balance=100.0)
        prompt = analyst.build_prompt(ctx)
        prompt_hash = sha256_of(prompt)
        prompt_preview = textwrap.shorten(prompt, width=PREVIEW_WIDTH, placeholder="...")

    return {
        "features_hash": features_hash,
        "signal_hash": signal_hash,
        "prompt_hash": prompt_hash,
        "retrieval_hash": None,
        "prompt_preview": prompt_preview,
        "versions": _versions(),
        "timing": {
            "features_runtime_ms": round(features_ms, 3),
            "strategy_runtime_ms": round(strategy_ms, 3),
        },
        "_snapshots": {
            "features": features_dump,
            "signal": signal_dump,
            "prompt": prompt,
        },
    }


def expected_payload(result: dict) -> dict:
    # The blessed-file shape: drop the ephemeral _snapshots and prompt full text.
    return {
        "features_hash": result["features_hash"],
        "signal_hash": result["signal_hash"],
        "prompt_hash": result["prompt_hash"],
        "retrieval_hash": result["retrieval_hash"],
        "prompt_preview": result["prompt_preview"],
        "versions": result["versions"],
        "timing": result["timing"],
    }


def load_case(name: str) -> dict:
    with open(os.path.join(CASES_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def load_expected(name: str) -> dict:
    with open(os.path.join(EXPECTED_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def write_expected(name: str, payload: dict) -> None:
    os.makedirs(EXPECTED_DIR, exist_ok=True)
    with open(os.path.join(EXPECTED_DIR, name), "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def list_cases() -> list[str]:
    if not os.path.isdir(CASES_DIR):
        return []
    return sorted(n for n in os.listdir(CASES_DIR) if n.endswith(".json"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/golden/test_runner.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/golden/golden_runner.py tests/golden/test_runner.py
git commit -m "feat: golden runner core (per-stage hashes, trade_id normalization)"
```

---

## Task 4: Runner CLI (--update / --explain)

**Files:**
- Modify: `tests/golden/golden_runner.py` (append CLI)
- Test: `tests/golden/test_runner_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/golden/test_runner_cli.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/golden/test_runner_cli.py -v`
Expected: FAIL — `AttributeError: module 'tests.golden.golden_runner' has no attribute 'print_explain'`.

- [ ] **Step 3: Append CLI + explain to golden_runner.py**

Append to `tests/golden/golden_runner.py`:

```python
def print_explain(case: dict, result: dict, expected: dict | None) -> None:
    snaps = result["_snapshots"]
    stages = [
        ("FEATURES", "features_hash", snaps["features"]),
        ("SIGNAL", "signal_hash", snaps["signal"]),
        ("PROMPT", "prompt_hash", snaps["prompt"]),
    ]
    print(f"=== {case['case_metadata'].get('description', '')} ===")
    for title, hkey, snap in stages:
        print(f"\n--- {title} ---")
        if snap is None:
            print("(no output for this stage)")
        elif title == "PROMPT":
            print(canonicalize(snap))  # canonical (string) form = exactly what is hashed
        else:
            print(canonicalize(snap))
        print(f"{hkey} = {result[hkey]}")
        if expected is not None and expected.get(hkey) != result[hkey]:
            print(f"  MISMATCH {hkey}: expected {expected.get(hkey)} got {result[hkey]}")
    print("\n--- timing ---")
    print(json.dumps(result["timing"]))


def _main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Golden replay runner")
    p.add_argument("--update", action="store_true", help="regenerate expected files")
    p.add_argument("--explain", metavar="CASE", help="print canonical snapshots + hashes")
    p.add_argument("cases", nargs="*", help="case filenames (default: all)")
    args = p.parse_args(argv)

    if args.explain:
        case = load_case(args.explain)
        result = run_case(case)
        try:
            expected = load_expected(args.explain)
        except FileNotFoundError:
            expected = None
        print_explain(case, result, expected)
        return 0

    targets = args.cases or list_cases()
    for name in targets:
        result = run_case(load_case(name))
        if args.update:
            write_expected(name, expected_payload(result))
            print(f"blessed {name}")
        else:
            print(f"{name}: features={result['features_hash'][:12]} "
                  f"signal={result['signal_hash'][:12]} "
                  f"prompt={(result['prompt_hash'] or 'null')[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/golden/test_runner_cli.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/golden/golden_runner.py tests/golden/test_runner_cli.py
git commit -m "feat: golden runner CLI (--update bless, --explain diagnostics)"
```

---

## Task 5: Seed fixtures + bless expected hashes

**Files:**
- Create: `tests/golden/_gen_fixtures.py`
- Create (generated): `tests/golden/replay_cases/{long_signal,high_vix_rejection,insufficient_bars_nan}.json`
- Create (generated): `tests/golden/expected_hashes/{...}.json`

- [ ] **Step 1: Write the fixture generator**

Create `tests/golden/_gen_fixtures.py`:

```python
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
```

- [ ] **Step 2: Generate the fixtures**

Run: `python -m tests.golden._gen_fixtures`
Expected: prints `wrote long_signal.json`, `wrote high_vix_rejection.json`, `wrote insufficient_bars_nan.json` with no assertion error.

If the `long_signal` assertion trips (RSI crept ≥ 70 on this machine's pandas_ta build), widen the dip in `_uptrend_bars` (e.g. `price -= 1.5`) and rerun — the periodic dip controls RSI.

- [ ] **Step 3: Bless the expected hashes**

Run: `python -m tests.golden.golden_runner --update`
Expected: prints `blessed long_signal.json`, `blessed high_vix_rejection.json`, `blessed insufficient_bars_nan.json`. Three files appear in `tests/golden/expected_hashes/`.

- [ ] **Step 4: Sanity-check a blessed file**

Run: `python -m tests.golden.golden_runner --explain high_vix_rejection.json`
Expected: SIGNAL stage shows the rejection dump; `prompt_hash = None`; no MISMATCH lines.

- [ ] **Step 5: Commit**

```bash
git add tests/golden/_gen_fixtures.py tests/golden/replay_cases tests/golden/expected_hashes
git commit -m "feat: golden seed fixtures + blessed expected hashes"
```

---

## Task 6: Parametrized golden test

**Files:**
- Create: `tests/golden/test_golden.py`

- [ ] **Step 1: Write the test**

Create `tests/golden/test_golden.py`:

```python
import pytest
from tests.golden import golden_runner as gr

CASES = gr.list_cases()


@pytest.mark.parametrize("name", CASES)
def test_golden_case_matches_expected(name):
    assert CASES, "no replay cases found — run python -m tests.golden._gen_fixtures"
    try:
        expected = gr.load_expected(name)
    except FileNotFoundError:
        pytest.fail(f"no expected file for {name} — run "
                    f"`python -m tests.golden.golden_runner --update {name}` to bless")
    result = gr.run_case(gr.load_case(name))

    assert result["features_hash"] == expected["features_hash"], f"{name}: features drift"
    assert result["signal_hash"] == expected["signal_hash"], f"{name}: signal drift"
    assert result["prompt_hash"] == expected["prompt_hash"], f"{name}: prompt drift"
    assert result["retrieval_hash"] == expected["retrieval_hash"]
    assert result["versions"] == expected["versions"], (
        f"{name}: version changed without re-bless — review and rerun --update")
```

- [ ] **Step 2: Run the golden suite**

Run: `pytest tests/golden/test_golden.py -v`
Expected: PASS — one parametrized case each for `long_signal`, `high_vix_rejection`, `insufficient_bars_nan`.

- [ ] **Step 3: Verify the lock actually catches drift**

Run: `python -m tests.golden.golden_runner --explain long_signal.json`
Confirm the printed `features_hash` equals the value in `tests/golden/expected_hashes/long_signal.json` (the harness compares the same canonical form it prints).

- [ ] **Step 4: Run the full suite (no regressions)**

Run: `pytest`
Expected: all prior tests + new golden tests green.

- [ ] **Step 5: Commit**

```bash
git add tests/golden/test_golden.py
git commit -m "feat: parametrized golden determinism assertions"
```

---

## Self-review notes

- **Spec coverage:** canonicalization (Task 1), `build_prompt_context`/`build_prompt` extraction (Task 2), per-stage hashes + `retrieval_hash` slot + `prompt_preview` + `timing` + `versions` incl. `canonical_schema_version` + trade_id normalization (Task 3), `--update`/`--explain` (Task 4), three seed cases (Task 5), parametrized assertions incl. version-bump lock (Task 6). All spec sections mapped.
- **Determinism hazards handled:** float noise (6dp round), NaN/Inf (string encoding), `\r\n` newlines (normalized), `trade_id=uuid4` (sentinel override), UTF-8 hashing (explicit), serialize-before-hash (`model_dump(mode="json")`).
- **Seams preserved:** `AIAnalyst.__new__` + class-level `_retriever=None` make `build_prompt_context`/`build_prompt` callable with no client and no memory header; existing `test_analyst.py` / `test_analyst_memory.py` stay green (verified in Task 2 Step 4).
- **Naming consistency:** `run_case`, `expected_payload`, `load_case`, `load_expected`, `write_expected`, `list_cases`, `print_explain`, `sha256_of`, `canonicalize` used identically across runner, CLI, and tests.
