# AI Trading Bot — Phase 1 Core Engine Design

**Date:** 2026-05-16
**Scope:** P1 Core Engine (paper trading, single strategy, Alpaca, Claude AI, Streamlit dashboard)
**Out of scope:** P2 backtesting platform, P3 live trading, P4 advanced intelligence

---

## 1. Architecture

### Approach
Pipeline monolith. Each module is a pure class with defined inputs/outputs. APScheduler drives a linear pipeline. No module is aware of others — data flows through `PipelineContext`, never sideways.

### Pipeline Flow

```
APScheduler (9:45 AM)
    │
    ▼
DataFetcher ──► FeaturePipeline ──► SpyTrendStrategy ──► AIAnalyst
                                                              │
                                                         RiskEngine
                                                              │
                                                       TradeValidator
                                                              │
                                         ┌────────────────────┴────────────────────┐
                                         ▼                                         ▼
                               TRADING_MODE=dry_run                    TRADING_MODE=paper_auto
                                  log only                               ExecutionEngine (Alpaca)
                                         │                                         │
                                         └────────────────────┬────────────────────┘
                                                              ▼
                                                    Logger (atomic) + .md report

Manual approval path (live_manual mode):
TradeValidator ──► pending_trades (DB) ──► FastAPI /approve/{trade_id} ──► ExecutionEngine
                                                    ▲
                                            Streamlit [APPROVE] button
```

### Trading Mode Flag

```python
TRADING_MODE = "dry_run"  # dry_run | paper_auto | live_manual | live_auto
```

P1 starts in `dry_run`. Progression: `dry_run` → `paper_auto` → `live_manual`.

---

## 2. Directory Structure

```
trading-bot/
├── app/
│   ├── config.py                   # TRADING_MODE, risk params, API config
│   ├── scheduler.py                # APScheduler job definitions
│   │
│   ├── pipeline/
│   │   ├── trading_pipeline.py     # Orchestrates full cycle
│   │   ├── context.py              # PipelineContext (runtime) + CycleRecord (DB)
│   │   └── models.py               # Shared pipeline enums
│   │
│   ├── models/
│   │   ├── market.py               # MarketData
│   │   ├── signals.py              # TradeSignal, TradeRejection
│   │   ├── ai.py                   # AIAnalysis
│   │   ├── risk.py                 # RiskDecision, ValidationResult
│   │   └── execution.py            # OrderResult, FillRecord
│   │
│   ├── prompts/
│   │   ├── system_prompt.txt       # Claude system role + authority context
│   │   └── market_analysis.txt     # Per-cycle analysis prompt template
│   │
│   ├── data/
│   │   └── fetcher.py              # Alpaca OHLCV + Finnhub news
│   │
│   ├── features/
│   │   └── pipeline.py             # Indicator computation → FeatureSet
│   │
│   ├── strategies/
│   │   └── spy_trend.py            # SPY Trend Following rules
│   │
│   ├── ai/
│   │   └── analyst.py              # Claude call, prompt injection, response parse
│   │
│   ├── risk/
│   │   └── engine.py               # Ordered kill switches + composite risk score
│   │
│   ├── validation/
│   │   └── validator.py            # Final pre-execution gate
│   │
│   ├── execution/
│   │   └── executor.py             # Alpaca paper orders, fill tracking
│   │
│   ├── persistence/
│   │   └── logger.py               # SQLite writes + .md report generation
│   │
│   └── dashboard/
│       └── app.py                  # Streamlit UI
│
├── config/
│   └── settings.yaml               # Non-secret config
│
├── data/
│   └── trading.db                  # SQLite
│
├── reports/
│   └── YYYY-MM-DD.md               # Daily report output
│
├── scripts/
│   ├── replay_day.py               # Replay full cycle from CycleRecord
│   └── seed_db.py                  # DB schema init
│
├── tests/
├── .env                            # API keys (never committed)
└── main.py                         # Entry point (FastAPI + APScheduler)
```

---

## 3. Data Layer + Feature Pipeline

### DataFetcher (`app/data/fetcher.py`)

Pulls from Alpaca (OHLCV bars, account info) and Finnhub (news sentiment). Returns typed `MarketData`.

**Fetched per cycle:**
- SPY daily + intraday bars (1m, 5m, 15m)
- Premarket data
- Relative volume vs 20-day average
- Finnhub news sentiment (last 24h)
- VIX level
- Bid/ask spread (fed into `FeatureSet` as `spread_proxy`)

Raw data cached to SQLite within the cycle to avoid redundant API calls.

### FeaturePipeline (`app/features/pipeline.py`)

Pure function: `MarketData` → `FeatureSet`. No external calls. All computed from cached OHLCV.

```python
@dataclass
class FeatureSet:
    # Identity
    symbol: str
    timeframe: str
    timestamp: datetime
    feature_version: str        # bump when indicator logic changes

    # Price
    price: float
    vwap: float

    # EMAs
    ema_9: float
    ema_20: float
    ema_50: float

    # Momentum
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float

    # Volatility
    atr: float
    vix: float

    # Volume
    relative_volume: float      # vs 20-day avg

    # Liquidity proxy
    spread_proxy: float         # (ask - bid) / mid

    # Context
    news_sentiment: float       # -1.0 to 1.0
```

**Key constraint:** Strategy engine consumes `FeatureSet` only — never raw price data.

---

## 4. Strategy Engine

### SPY Trend Following (`app/strategies/spy_trend.py`)

Pure function: `FeatureSet` → `TradeSignal | TradeRejection`. Deterministic. No AI calls.

**Entry conditions (all must pass):**
```python
conditions = {
    "spy_above_ema20":   features.price > features.ema_20,
    "ema9_above_ema20":  features.ema_9 > features.ema_20,
    "volume_confirmed":  features.relative_volume > 1.2,
    "vix_acceptable":    features.vix < 25,
    "not_overbought":    features.rsi < 70,
}
```

If all conditions pass → `TradeSignal`. Otherwise → `TradeRejection` with explicit failed conditions.

**Confidence scoring (monotonic weighted sum):**
```python
strategy_confidence = (
    ema_strength(features)      * 0.30 +
    volume_strength(features)   * 0.20 +
    trend_strength(features)    * 0.30 +
    volatility_score(features)  * 0.20
)
# filters = binary (pass/fail); scoring = continuous (0.0–1.0)
# more condition strength always = higher confidence
```

**Output models:**
```python
class TradeSignal(BaseModel):
    type: Literal["SIGNAL"]
    symbol: str
    strategy: str
    direction: Literal["long", "short"]
    strategy_confidence: float
    entry_price: float
    stop_loss: float            # entry - (atr * 2)
    take_profit: float          # entry + (atr * 3)

class TradeRejection(BaseModel):
    type: Literal["REJECTION"]
    symbol: str
    strategy: str
    reasons: list[str]          # e.g. ["VIX too high", "RSI overbought"]
    strategy_confidence: float
```

**Extensibility:** `strategies/` is a registry. P2 adds `mean_reversion.py`, `orb.py` as drop-in modules.

---

## 5. AI Analyst

### Claude call (`app/ai/analyst.py`)

One call per cycle. Input: `FeatureSet + TradeSignal | TradeRejection`. Output: `AIAnalysis`.

**Code-enforced non-override rule:**
```python
if isinstance(signal, TradeRejection):
    # skip LLM entirely — strategy already rejected
    return AIAnalysis(decision="REJECT", ai_confidence=0.0, ...)
```
Claude cannot approve a rejected signal. Enforced in code, not only in prompt.

**Prompt structure** (template in `app/prompts/market_analysis.txt`):
```
STRATEGY AUTHORITY:
- The strategy engine is deterministic and must NOT be overridden.
- You are an annotator and risk evaluator only.
- If TradeSignal type is REJECTION, you MUST output decision=REJECT.

RULES (non-negotiable):
- Max 1 trade per day
- Max position size: 20% of capital
- Stop loss required on every trade
- If ai_confidence < 0.70, output REJECT
- Avoid earnings within 3 days
- TRADING_MODE: {trading_mode}

MARKET DATA:
{feature_set_json}

STRATEGY SIGNAL:
{trade_signal_json}

Return JSON matching this schema:
{schema}
```

**Output model:**
```python
class AIAnalysis(BaseModel):
    decision: Literal["APPROVE", "REJECT", "REDUCE_CONFIDENCE", "NO_TRADE"]
    ai_confidence: float            # separate from strategy_confidence
    regime: Literal["bullish", "bearish", "neutral", "uncertain"]
    reasoning: str
    risk_factors: list[str]
    no_trade_reasons: list[str]
    raw_prompt: str                 # stored verbatim for replay
    raw_response: str               # stored verbatim for replay
    model_version: str              # e.g. "claude-sonnet-4-6"
    prompt_version: str             # from prompts/ file header
```

**Parse failure handling:** If Claude response fails JSON parse → abort trading decision, mark `AI_ANALYSIS=FAILED`, continue pipeline (logging + report still write). No trade executed on parse failure.

---

## 6. Risk Engine + Trade Validator

### RiskEngine (`app/risk/engine.py`)

Input: `TradeSignal + AIAnalysis + AccountState` → `RiskDecision`.

**Ordered evaluation (stop at first hit):**

```python
# 1. KILL SWITCHES (absolute stop — trading disabled until manual reset)
if drawdown_today > 0.10:              raise KillSwitch("drawdown > 10%")
if consecutive_losses >= 3:            raise KillSwitch("3 consecutive losses")
if kill_switch_state.active:           raise KillSwitch("manual kill active")

# 2. HARD BLOCKS (reject this trade, trading remains enabled)
if account_balance < MIN_BALANCE:      return BLOCKED("balance below floor")
if position_size > balance * 0.20:     return BLOCKED("position too large")
if daily_trade_count >= MAX_TRADES:    return BLOCKED("daily limit reached")
if stop_loss_missing:                  return BLOCKED("no stop loss")
if not market_hours_valid():           return BLOCKED("outside market hours")
if earnings_within_3_days("SPY"):      return BLOCKED("earnings proximity")
if cooldown_active():                  return BLOCKED("cooldown window active")
if open_position_in_symbol():          return BLOCKED("duplicate position")

# 3. COMPOSITE RISK SCORE
risk_score = (
    0.40 * signal.strategy_confidence +
    0.30 * analysis.ai_confidence +
    0.30 * volatility_score(features)
)
if risk_score < RISK_SCORE_THRESHOLD:  return BLOCKED(f"risk_score {risk_score:.2f} below threshold")

return APPROVED(risk_score=risk_score)
```

Kill switch state persisted to SQLite. Survives restarts. Manual reset required (dashboard or CLI).

**Cooldown:** 30-minute window after any trade. No direction-flip within same session.

**Market hours:** Uses `pandas_market_calendars` NYSE calendar — handles federal holidays, early closes, DST transitions.

### TradeValidator (`app/validation/validator.py`)

Final gate. Checks live market conditions risk engine cannot see:

```python
if spread_pct > MAX_SPREAD_PCT:        return FAIL("spread too wide")
if buying_power < order_value:         return FAIL("insufficient buying power")
if TRADING_MODE == "dry_run":          return FAIL("dry_run mode — no execution")
if TRADING_MODE == "live_manual":
    write_pending_trade(signal)
    return FAIL("awaiting manual approval")
return PASS()
```

---

## 7. Execution Engine

### Executor (`app/execution/executor.py`)

Input: `ValidationResult(PASS) + TradeSignal + TRADING_MODE` → places order via Alpaca SDK.

**Idempotency:**
- Every trade assigned `trade_id: UUID` at `TradeSignal` creation
- Every execution attempt assigned `execution_id: UUID`
- DB constraint: one execution per `trade_id`. Duplicate call → immediate reject.

**Order placement (Alpaca bracket orders preferred):**
```python
order = alpaca.submit_order(
    symbol="SPY",
    qty=compute_shares(position_size, price),
    side="buy",
    type="market",
    time_in_force="day",
    order_class="bracket",          # SL + TP bundled
    stop_loss={"stop_price": signal.stop_loss},
    take_profit={"limit_price": signal.take_profit},
)
# position_side="long" annotated on all child orders (future short-safe)
```

If bracket not available → market order placed first, then stop-loss order immediately after fill. If stop-loss placement fails → position closed immediately, logged as critical error.

**Fill tracking:**
```python
# Poll with timeout
fill = poll_fill(order_id, max_wait_seconds=30)
if fill is None:
    mark_order_stale(order_id)
    reconcile_with_broker(order_id)
    return

record_fill(
    fill_price=fill.price,
    fill_time=fill.timestamp,
    slippage=fill.price - signal.entry_price,
    broker_state="filled",          # submitted|filled|partial_fill|canceled|reconciled
)
```

**Execution fingerprint:**
```python
execution_fingerprint = sha256(
    json(signal) + json(features) + json(ai_analysis)
).hexdigest()
```
Stored on every execution. Enables replay reconstruction.

**FastAPI** (`main.py`): single endpoint `POST /approve/{trade_id}` for `live_manual` mode. Re-runs risk + validation before executing. Idempotency enforced.

---

## 8. Persistence + Reporting

### SQLite Schema

```sql
-- Versioned artifacts (all tables include these columns)
-- pipeline_version, strategy_version, ai_prompt_version, model_version

market_snapshots     -- raw OHLCV per cycle
feature_sets         -- FeatureSet JSON + feature_version
trade_signals        -- TradeSignal or TradeRejection JSON
ai_analyses          -- AIAnalysis JSON, raw_prompt, raw_response
risk_decisions       -- RiskDecision + kill switch state
trade_executions     -- fill_price, slippage, broker_state, execution_fingerprint
positions            -- open/closed positions + PnL
daily_summary        -- win_rate, drawdown, P&L, strategy_performance (JSON)

pending_trades       -- lifecycle: PENDING_APPROVAL|APPROVED|REJECTED|EXPIRED|EXECUTED|FAILED
kill_switch_state    -- persisted kill switch (manual reset required)
```

### Logger (`app/persistence/logger.py`)

One atomic SQLite transaction per cycle. Boundary: `DataFetcher → Logger`. Either full cycle commits or none. No partial state.

Writes `PipelineContext` (runtime) by serializing to `CycleRecord` (primitives + JSON blobs). No live Python objects stored.

### Daily `.md` Report (`reports/YYYY-MM-DD.md`)

```markdown
# Trading Report — 2026-05-16

## Market Assessment
Regime: Bullish | VIX: 18.2 | SPY: +0.4% | Spread: 0.01%

## Strategy Signal
SPY Trend Following: LONG | strategy_confidence: 0.74

## AI Analysis
Decision: APPROVE | ai_confidence: 0.81 | Model: claude-sonnet-4-6
Reasoning: [stored in DB — fetch with replay_day.py]
Risk factors: elevated pre-market volume, FOMC in 3 days

## Trade
Status: EXECUTED (paper) | Entry: $521.20 | SL: $517.00 | TP: $529.00
Slippage: +$0.03 | Fingerprint: abc123...

## Risk Engine
risk_score: 0.76 | Kill switches: INACTIVE | Cooldown: CLEAR

## Session P&L
Today: +$3.20 | 7-day drawdown: 1.2% | Consecutive losses: 0

## Strategy Performance (7-day)
spy_trend_following: { win_rate: 0.60, profit_factor: 1.4, trades: 5 }
```

---

## 9. Streamlit Dashboard

Single-page app (`app/dashboard/app.py`). Reads SQLite directly (read-only). Calls FastAPI only for trade approval.

**Header (always visible):**
```
Account: $100.00 paper  |  Mode: DRY_RUN  |  Last Updated: 09:45:12 AM  |  Data Age: 3s
```

**Sections:**
1. **Pending Approval** — shown only in `live_manual` mode
   - Signal details (symbol, direction, entry, SL, TP, both confidence scores)
   - Risk engine status badge: `✓ Risk Engine approved` or `⛔ Blocked — cannot approve`
   - `[APPROVE]` / `[REJECT]` buttons (POST to FastAPI)

2. **Today's Cycle** — regime, risk score, AI reasoning (lazy-loaded on expand)

3. **Active Positions** — open positions with unrealized P&L

4. **Trade History** — last 50 trades (paginated, lazy load older)

5. **Performance** — rolling 7-day + last-50-trades toggle
   - Win rate, profit factor, max drawdown, consecutive losses

Auto-refresh every 60 seconds during market hours via `st.rerun()`.

**Key constraint:** Streamlit contains zero business logic. State only. All mutations go through FastAPI.

---

## 10. Scheduler + Pipeline Orchestration

### APScheduler Jobs (`app/scheduler.py`)

All jobs configured with `max_instances=1, coalesce=True, misfire_grace_time=30` to prevent overlapping cycles.

Market hours enforced via `pandas_market_calendars` NYSE calendar before any job executes.

```python
scheduler.add_job(market_assessment_job,  "cron", hour=9,  minute=45,
                  max_instances=1, coalesce=True, misfire_grace_time=30)
scheduler.add_job(trade_decision_job,     "cron", hour=10, minute=0,
                  max_instances=1, coalesce=True, misfire_grace_time=30)
scheduler.add_job(position_monitor_job,   "interval", minutes=5,
                  max_instances=1, coalesce=True, misfire_grace_time=30)
scheduler.add_job(close_positions_job,    "cron", hour=15, minute=55,
                  max_instances=1, coalesce=True, misfire_grace_time=30)
scheduler.add_job(daily_report_job,       "cron", hour=16, minute=10,
                  max_instances=1, coalesce=True, misfire_grace_time=30)
```

### Pipeline Coordinator (`app/pipeline/trading_pipeline.py`)

```python
class TradingPipeline:
    def run_assessment(self) -> PipelineContext:
        ctx = PipelineContext(cycle_id=uuid4(), started_at=now())
        ctx.market_data  = DataFetcher.fetch(ctx)
        ctx.features     = FeaturePipeline.compute(ctx.market_data)
        ctx.signal       = SpyTrendStrategy.evaluate(ctx.features)
        ctx.ai_analysis  = AIAnalyst.analyze(ctx.signal, ctx.features)
        ctx.risk         = RiskEngine.evaluate(ctx)
        ctx.validation   = TradeValidator.validate(ctx)
        Logger.write_cycle(ctx)          # atomic — PipelineContext → CycleRecord
        ReportGenerator.update_md(ctx)
        return ctx
```

**`PipelineContext`** — runtime only. Passed by reference, immutable after each stage completes.

**`CycleRecord`** — DB-persisted. Primitives + JSON blobs. No live objects. Supports full replay.

### Replay (`scripts/replay_day.py`)

Loads historical `CycleRecord` from SQLite. Supports mode override:

```python
REPLAY_MODE = "full_pipeline"
# full_pipeline | feature_only | strategy_only | ai_only
```

- `feature_only` — recompute features from raw market data, no LLM calls
- `strategy_only` — replay strategy logic against stored features
- `ai_only` — re-run Claude prompt against stored signal (prompt tuning)
- `full_pipeline` — full re-run from stored raw data

---

## 11. Configuration Defaults

| Constant | Value | Notes |
|---|---|---|
| `TRADING_MODE` | `dry_run` | Start here; promote manually |
| `STARTING_CAPITAL` | `$100.00` | Paper account |
| `MIN_BALANCE_THRESHOLD` | `$10.00` | Hard floor; kill trading below this |
| `MAX_POSITION_SIZE_PCT` | `0.20` | 20% of account per trade |
| `MAX_DAILY_TRADES` | `1` | Increase to 3 only after stable paper results |
| `MAX_SPREAD_PCT` | `0.05%` | Reject if bid/ask spread > 0.05% of mid |
| `RISK_SCORE_THRESHOLD` | `0.65` | Composite score floor for execution |
| `AI_CONFIDENCE_THRESHOLD` | `0.70` | Minimum `ai_confidence` to APPROVE |
| `MAX_DRAWDOWN_PCT` | `0.10` | 10% daily drawdown → kill switch |
| `CONSECUTIVE_LOSS_LIMIT` | `3` | 3 losses in a row → kill switch |
| `VIX_MAX` | `25` | Above this → strategy rejects |
| `RSI_MAX_LONG` | `70` | Overbought threshold for long entries |
| `RELATIVE_VOLUME_MIN` | `1.2` | Volume confirmation floor |
| `ATR_STOP_MULTIPLIER` | `2.0` | Stop loss = entry - (ATR * 2) |
| `ATR_TARGET_MULTIPLIER` | `3.0` | Take profit = entry + (ATR * 3) |
| `COOLDOWN_MINUTES` | `30` | Wait after any trade before next signal |
| `EARNINGS_PROXIMITY_DAYS` | `3` | Avoid SPY within 3 days of earnings |
| `FILL_POLL_TIMEOUT_SECONDS` | `30` | Stale after this; reconcile with broker |
| `DASHBOARD_REFRESH_SECONDS` | `60` | Streamlit auto-rerun interval |
| `TRADE_HISTORY_PAGE_SIZE` | `50` | Default rows shown in trade history |

All values live in `config/settings.yaml` (non-secrets) or `.env` (API keys). Override via environment variables.

---

## 12. Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.12+ | |
| API framework | FastAPI | Single endpoint for approval |
| Scheduler | APScheduler | Embedded in main process |
| AI | Claude API (Sonnet) | One call per cycle |
| Broker | Alpaca Paper Trading | Bracket orders |
| Market data | Alpaca + Finnhub | OHLCV + news |
| Database | SQLite | MVP; PostgreSQL in P2+ |
| ORM | SQLAlchemy | |
| Validation | Pydantic v2 | All models |
| Dashboard | Streamlit | Read-only SQLite |
| Market calendar | pandas_market_calendars | NYSE holidays + early closes |
| Config | python-dotenv + PyYAML | `.env` + `settings.yaml` |

---

## 13. Key Invariants

1. **Strategy authority:** Strategy produces signals. AI annotates. Risk engine decides. These roles never cross.
2. **AI cannot override:** `TradeRejection` → skip LLM, force `REJECT`. Enforced in code.
3. **Risk engine ordered:** KILL SWITCHES → HARD BLOCKS → COMPOSITE SCORE. Never skip a tier.
4. **Idempotency:** One execution per `trade_id`. Enforced at DB level.
5. **Atomic cycles:** Full pipeline commits or nothing. No partial state.
6. **Versioned artifacts:** Every stored record carries `pipeline_version`, `strategy_version`, `ai_prompt_version`, `model_version`.
7. **Replay-first logging:** `raw_prompt` + `raw_response` stored verbatim. `execution_fingerprint` on every trade.
8. **Dashboard is passive:** Zero business logic in Streamlit. All mutations through FastAPI.

---

## 14. Out of Scope for P1

- Backtesting (P2)
- Multi-strategy (P2)
- Live capital (P3)
- Fully autonomous execution (P3/P4)
- Multi-agent workflows (P4)
- Reinforcement learning (P4)
- PostgreSQL migration (P2+)
- Options, crypto, leveraged ETFs (explicitly excluded)
