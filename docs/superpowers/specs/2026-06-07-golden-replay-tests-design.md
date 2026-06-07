# Golden Replay / Determinism Tests — Design

**Date:** 2026-06-07
**Status:** Approved (design)
**Scope:** ai-trading-bot — deterministic regression harness for the market-data → features → signal → AI-prompt path.

## Goal

Lock correctness of the deterministic pipeline. Given a frozen `MarketData` input, the
pipeline must produce byte-stable per-stage outputs across refactors, dependency bumps, and
machine differences. Any unintended change in feature math, strategy logic, or prompt
construction breaks a golden test loudly.

This is a **regression lock**, not a behavioral spec. It does not assert "the strategy should
go long here" — it asserts "the pipeline produces *exactly this* output for *this* input, and
if that changes you must consciously re-bless it."

## Determinism boundary

The pipeline is deterministic up to and including **prompt construction**. The LLM call itself
is non-deterministic and is explicitly OUT of the hashed boundary.

```
MarketData ─► FeaturePipeline.compute ─► FeatureSet ─► SpyTrendStrategy.evaluate ─► signal/rejection ─► AIAnalyst.build_prompt ─► prompt string
   (input)         [features_hash]                         [signal_hash]                              [prompt_hash]
```

We hash the *assembled prompt string*, which verifies prompt construction without coupling the
test to LLM output. No network call is made in golden tests.

## Why hashing (not value-equality)

`FeaturePipeline.compute` emits floats from `pandas_ta_classic` (ema/rsi/macd/atr/vwap/…).
Floats vary in the low-order bits across BLAS builds, numpy versions, and platforms. Storing
full-precision expected values would make tests flaky. Instead we **round to 6 decimal places**,
canonicalize to a stable JSON form, and SHA256 it. Rounding absorbs benign float noise while
still catching any change a human would consider meaningful.

## Layout

```
tests/golden/
  replay_cases/            # input fixtures (committed JSON)
    long_signal.json
    high_vix_rejection.json
    insufficient_bars_nan.json
  expected_hashes/         # blessed outputs (committed JSON)
    long_signal.json
    high_vix_rejection.json
    insufficient_bars_nan.json
  golden_runner.py         # replay engine + canonicalizer + CLI
  test_golden.py           # pytest: one parametrized test per case
```

## Replay-case schema

Each case wraps the frozen input in a metadata envelope:

```json
{
  "case_metadata": {
    "description": "SPY uptrend, all gates clear, expect long signal",
    "created_at": "2026-06-07T00:00:00",
    "source": "synthetic | captured:<date>",
    "expected_behavior": "long signal generated, prompt built"
  },
  "market_data": { ... MarketData.model_dump(mode="json") ... }
}
```

`market_data` is loaded via `MarketData(**case["market_data"])` so the fixture is validated by
the same Pydantic model the live pipeline uses.

## Expected-hashes schema

```json
{
  "features_hash": "<sha256>",
  "signal_hash": "<sha256>",
  "prompt_hash": "<sha256|null>",
  "retrieval_hash": null,
  "prompt_preview": "<first ~300 chars of assembled prompt, human-readable>",
  "versions": {
    "pipeline": "1.0.0",
    "strategy": "1.0.0",
    "prompt": "1.0.0"
  },
  "timing": {
    "features_runtime_ms": 0.0,
    "strategy_runtime_ms": 0.0
  }
}
```

- `prompt_hash` is `null` when the strategy returns a `TradeRejection` (no prompt built).
- `retrieval_hash` is reserved (`null`) for the future memory-retrieval stage — schema slot
  added now so adding the stage later does not reshape every expected file.
- `prompt_preview` is diagnostic only — NOT hashed, NOT asserted. It makes diffs readable.
- `timing` is diagnostic only — NOT hashed, NOT asserted. Captures wall-clock per stage to
  surface accidental perf regressions during `--explain`.
- `versions` ARE asserted: a version bump without a re-bless is a hard failure, forcing the
  author to acknowledge the contract changed.

## Canonicalization

`canonicalize(obj) -> str` produces stable JSON, then `sha256(...)`:

1. **Serialize first.** Always operate on `model.model_dump(mode="json")`, never raw object
   attributes. Pydantic's json mode already normalizes datetimes/enums/UUIDs consistently with
   production serialization.
2. **Numbers:** round every `float` to 6 dp. `NaN` → `"NaN"`, `+Inf` → `"Infinity"`,
   `-Inf` → `"-Infinity"` (JSON has no native form). Coerce `Decimal` → rounded float.
   Coerce numpy scalar types (`numpy.floating`, `numpy.integer`, `numpy.bool_`) → native Python
   primitive before rounding.
3. **datetime:** ISO-8601 string (already handled by `mode="json"`; the canonicalizer also
   guards any stray `datetime` defensively).
4. **Ordering:** dicts → keys sorted recursively. Lists → order preserved exactly (order is
   semantically meaningful, e.g. OHLC bar sequence).
5. **Dump:** `json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)`.
6. **Hash:** `hashlib.sha256(s.encode()).hexdigest()`.

The same `canonicalize` feeds both the hash and the `--explain` snapshots, so what you inspect
is exactly what gets hashed.

## Runner

`golden_runner.py` exposes:

- `run_case(case_dict) -> dict` — executes the pipeline, returns
  `{features_hash, signal_hash, prompt_hash, retrieval_hash, prompt_preview, versions, timing,
  canonical_snapshots}`.
- `load_case(path)`, `load_expected(path)`.
- CLI:
  - `python -m tests.golden.golden_runner --update [case…]` — regenerate expected files
    (bless). Writes hashes, `prompt_preview`, current versions; preserves/refreshes `timing`.
  - `python -m tests.golden.golden_runner --explain <case>` — print, per stage: canonical JSON
    snapshot, rounded values, the stage hash, and (when an expected file exists) a field-level
    diff between current and blessed canonical forms. This is the debugging path when a test
    fails: it tells you *which field* moved, not just "hash mismatch".

`run_case` flow:
1. `md = MarketData(**case["market_data"])`
2. time `features = FeaturePipeline().compute(md)` → `features_runtime_ms`,
   `features_hash = sha(canonicalize(features.model_dump(mode="json")))`
3. time `result = SpyTrendStrategy().evaluate(features, …)` → `strategy_runtime_ms`,
   `signal_hash = sha(canonicalize(result.model_dump(mode="json")))`
4. if `result` is a `TradeSignal`:
   `ctx = analyst.build_prompt_context(signal, features, account_balance=100.0)`
   `prompt = analyst.build_prompt(ctx)`
   `prompt_hash = sha(canonicalize(prompt))`, `prompt_preview = prompt[:300]`
   else `prompt_hash = None`, `prompt_preview = None`.

No `AIAnalyst` LLM client is constructed — only the pure prompt-building methods are called.

## Production change: extract prompt building

Currently `AIAnalyst.analyze()` builds the prompt inline (4-kwarg `.format(...)` on
`market_analysis.txt`, then concatenates the optional memory block) and immediately calls the
model. Golden tests need prompt construction callable WITHOUT a client and WITHOUT a network
call. Split into two pure methods:

- `build_prompt_context(self, signal, features, account_balance=100.0) -> dict` — assembles the
  values that flow into the template (the 4 format kwargs + any memory block string). Pure, no
  client, no I/O.
- `build_prompt(self, ctx: dict) -> str` — formats `market_analysis.txt` with `ctx` and returns
  the final assembled prompt string (memory block concatenated under its header when present).

`analyze()` is refactored to call `build_prompt_context` → `build_prompt`, then hand the string
to `_call_model`. Behavior is unchanged: with memory flags off, the produced prompt is
byte-identical to today (guarded by existing `tests/ai/test_analyst.py` and
`test_analyst_memory.py`). The `__new__` / `self._client` test seam and class-level
`_gateway=_retriever=_summarizer=None` defaults are preserved.

Splitting context-assembly from formatting keeps each method single-purpose: the runner can
hash the context separately later if needed, and prompt-format changes stay isolated from
context-gathering changes.

## Seed cases

1. **long_signal** — clean SPY uptrend, all gates pass → `TradeSignal`, prompt built. Exercises
   the full happy path including prompt_hash.
2. **high_vix_rejection** — VIX above `vix_max` → `TradeRejection`. Exercises rejection path,
   `prompt_hash = null`.
3. **insufficient_bars_nan** — too few bars so indicators are `NaN`. Exercises the
   `_last → float("nan")` edge and confirms NaN canonicalizes stably (`"NaN"`).

Later (follow-up, not this pass): borderline-confidence, malformed/partial market data, spread
violation, high-volatility chop, and a retrieval-enabled case once the memory stage is hashed.

## Testing

`test_golden.py` parametrizes over every case in `replay_cases/`:
- assert `features_hash`, `signal_hash`, `prompt_hash` match the blessed expected file.
- assert `versions` match (forces conscious re-bless on version bump).
- do NOT assert `prompt_preview` or `timing` (diagnostic only).
- a missing expected file fails with a clear "run --update to bless" message.

## Verification

1. `pytest tests/golden` green on seed cases.
2. Determinism: run the suite twice → identical results; run `run_case` twice on one case →
   identical hashes.
3. `--update` then `pytest` → green (round-trip bless).
4. `--explain <case>` prints readable per-stage canonical JSON + hashes + diff.
5. Existing `tests/ai/test_analyst.py` and `test_analyst_memory.py` stay green after the
   `build_prompt` extraction (zero prompt-construction regression with flags off).
6. Full suite (`pytest`) stays green.

## Out of scope (this pass)

- Hashing the memory-retrieval stage (`retrieval_hash` slot reserved, left `null`).
- Capturing real market-data fixtures from live APIs (seed cases are synthetic).
- Asserting timing thresholds (timing captured for inspection only).
