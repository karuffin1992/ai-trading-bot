# AI Trading Bot

An automated SPY trend-following trading bot that runs a deterministic strategy, annotates each signal with Claude AI risk analysis, and executes paper trades through Alpaca. It runs on a market-hours schedule, persists every decision to SQLite for full replay, and exposes a Streamlit dashboard plus a FastAPI approval endpoint.

> **Phase 1 (current):** paper trading only, single strategy, starts in `dry_run` mode. Backtesting, multi-strategy, and live capital are out of scope.

---

## How it works

The bot is a **pipeline monolith**. Each stage is a pure class with defined inputs and outputs. Data flows one direction through a `PipelineContext` — no module is aware of any other. APScheduler drives the pipeline on a daily cron.

```
APScheduler (9:45 AM ET)
    │
    ▼
DataFetcher ──► FeaturePipeline ──► SpyTrendStrategy ──► AIAnalyst
   (Alpaca       (indicators →        (rules →            (Claude:
    + Finnhub)     FeatureSet)         TradeSignal)         annotate/reject)
                                                              │
                                                         RiskEngine        (kill switches → hard blocks → composite score)
                                                              │
                                                       TradeValidator      (spread, buying power, mode gate)
                                                              │
                            ┌─────────────────────────────────┴───────────────────────────────┐
                            ▼                                                                   ▼
                    TRADING_MODE=dry_run                                          TRADING_MODE=paper_auto
                       log only                                                ExecutionEngine → Alpaca (bracket order)
                            │                                                                   │
                            └─────────────────────────────────┬───────────────────────────────┘
                                                              ▼
                                              Logger (atomic SQLite) + reports/YYYY-MM-DD.md
```

**Manual-approval path** (`live_manual` mode): the validator writes a pending trade to the DB, you approve it from the dashboard, which POSTs to FastAPI `/approve/{trade_id}`. Risk and validation re-run before execution.

### Key invariants

- **Strategy authority** — the strategy produces signals, AI only annotates, the risk engine decides. If the strategy emits a `TradeRejection`, the LLM is skipped entirely and the decision is forced to `REJECT` in code.
- **Ordered risk engine** — kill switches → hard blocks → composite score, never skipping a tier. The kill switch state persists across restarts and requires a manual reset.
- **Atomic cycles** — the full pipeline commits in one SQLite transaction or nothing does.
- **Replay-first** — `raw_prompt`, `raw_response`, and an `execution_fingerprint` are stored on every cycle, so any day can be re-run with `scripts/replay_day.py`.

### Trading modes

Set in `config/settings.yaml` (`trading.mode`) or via the `TRADING_MODE` env var. Promote manually as results stabilize:

| Mode | Behavior |
|---|---|
| `dry_run` | Run the full pipeline, log everything, **execute nothing** (default) |
| `paper_auto` | Auto-execute paper orders through Alpaca |
| `live_manual` | Queue trades for manual approval via the dashboard/API |
| `live_auto` | Fully autonomous (out of scope for P1) |

---

## Project layout

```
app/
├── config.py                 # Settings (env + YAML merge)
├── scheduler.py              # APScheduler jobs, NYSE market-hours guard
├── pipeline/
│   └── trading_pipeline.py   # Orchestrates a full cycle
├── data/fetcher.py           # Alpaca OHLCV + Finnhub news → MarketData
├── features/pipeline.py      # MarketData → FeatureSet (indicators)
├── strategies/spy_trend.py   # FeatureSet → TradeSignal | TradeRejection
├── ai/analyst.py             # Claude call, prompt build, response parse
├── risk/engine.py            # Kill switches → hard blocks → risk score
├── validation/validator.py   # Final pre-execution gate
├── execution/executor.py     # Alpaca bracket orders, fill tracking
├── persistence/              # SQLite writes, daily summary, .md reports
├── dashboard/app.py          # Streamlit UI (read-only DB + approval calls)
└── models/                   # Pydantic models for each stage
config/settings.yaml          # Non-secret config
scripts/
├── seed_db.py                # Create DB schema + seed kill switch
└── replay_day.py             # Replay a stored cycle for debugging
main.py                       # Entry point: FastAPI app + scheduler
```

---

## Running locally

### Prerequisites

- **Python 3.12+**
- API keys for **Alpaca** (paper account), **Finnhub**, and **Anthropic** (Claude)

### 1. Set up the environment

```powershell
# from the project root
python -m venv .venv
.venv\Scripts\Activate.ps1          # PowerShell
pip install -e ".[dev]"
```

### 2. Configure secrets

Copy the example env file and fill in your keys:

```powershell
Copy-Item .env.example .env
```

```ini
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
FINNHUB_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

Non-secret settings (mode, capital, risk thresholds) live in `config/settings.yaml`. The default mode is `dry_run`, so no real or paper orders are placed until you change it.

### 3. Initialize the database

Creates `data/trading.db`, the `reports/` directory, and seeds the kill-switch row:

```powershell
python scripts/seed_db.py
```

### 4. Start the app (API + scheduler)

```powershell
uvicorn main:app --reload
```

This starts FastAPI and the APScheduler background jobs. Health check:

```
GET http://localhost:8000/health   →  {"status": "ok", "mode": "dry_run"}
```

The scheduler runs on US/Eastern time and **skips all jobs when the NYSE is closed**, so outside market hours nothing trades — that's expected. Scheduled jobs:

| Time (ET) | Job |
|---|---|
| 9:45 AM | Run the assessment pipeline (one cycle) |
| every 5 min | Sync open positions |
| 3:55 PM | Flatten all positions (EOD) |
| 4:10 PM | Compute the daily summary |

### 5. Start the dashboard (separate terminal)

```powershell
streamlit run app/dashboard/app.py
```

The dashboard reads SQLite directly (read-only) and shows mode, P&L, kill-switch state, positions, and trade history. In `live_manual` mode it shows pending trades with **Approve / Reject** buttons that call the FastAPI endpoints.

---

## Useful commands

```powershell
# Run the test suite
pytest

# Replay a stored cycle (debugging / prompt tuning)
python scripts/replay_day.py --date 2026-05-16
python scripts/replay_day.py --date 2026-05-16 --mode ai_only   # re-run Claude only
```

Replay modes: `full_pipeline`, `feature_only`, `strategy_only`, `ai_only`.

---

## Tech stack

Python 3.12 · FastAPI · APScheduler · Claude API (Sonnet) · Alpaca paper trading · Finnhub · SQLAlchemy + SQLite · Pydantic v2 · Streamlit · pandas / pandas-ta · pandas-market-calendars.

See `docs/superpowers/specs/2026-05-16-trading-bot-p1-design.md` for the full design spec.
