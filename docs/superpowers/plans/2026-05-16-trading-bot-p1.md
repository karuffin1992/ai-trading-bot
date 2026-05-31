# AI Trading Bot — Phase 1 Core Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a paper trading bot that runs a daily SPY Trend Following cycle with Claude AI analysis, strict risk controls, optional manual approval, and a Streamlit dashboard.

**Architecture:** Pipeline monolith driven by APScheduler. Each stage (DataFetcher → FeaturePipeline → SpyTrendStrategy → AIAnalyst → RiskEngine → TradeValidator → ExecutionEngine → Logger) is a pure class with typed Pydantic I/O. `PipelineContext` carries runtime state; `CycleRecord` persists to SQLite. FastAPI exposes `POST /approve/{trade_id}`. Streamlit reads SQLite directly.

**Tech Stack:** Python 3.12, FastAPI, APScheduler 3.x, SQLAlchemy 2.x, Pydantic v2, pydantic-settings, anthropic SDK, alpaca-py, finnhub-python, yfinance, pandas, pandas-ta, pandas-market-calendars, Streamlit, pytest, pytest-mock, freezegun

---

## File Map

| Task | Files |
|------|-------|
| 1 | `pyproject.toml`, `.env.example`, `.gitignore`, `config/settings.yaml`, `app/config.py`, `main.py`, `tests/conftest.py` |
| 2 | `app/models/market.py`, `app/models/signals.py`, `app/models/ai.py`, `app/models/risk.py`, `app/models/execution.py`, `app/pipeline/context.py`, `app/pipeline/models.py`, `tests/test_models.py` |
| 3 | `app/persistence/db.py`, `scripts/seed_db.py`, `tests/persistence/test_db.py` |
| 4 | `app/data/fetcher.py`, `tests/data/test_fetcher.py` |
| 5 | `app/features/pipeline.py`, `tests/features/test_pipeline.py` |
| 6 | `app/strategies/spy_trend.py`, `tests/strategies/test_spy_trend.py` |
| 7 | `app/prompts/system_prompt.txt`, `app/prompts/market_analysis.txt`, `app/ai/analyst.py`, `tests/ai/test_analyst.py` |
| 8 | `app/risk/engine.py`, `tests/risk/test_engine.py` |
| 9 | `app/validation/validator.py`, `tests/validation/test_validator.py` |
| 10 | `app/execution/executor.py`, `tests/execution/test_executor.py` |
| 11 | `app/persistence/logger.py`, `tests/persistence/test_logger.py` |
| 12 | `app/pipeline/trading_pipeline.py`, `tests/pipeline/test_trading_pipeline.py` |
| 13 | `app/scheduler.py`, `main.py` (update) |
| 14 | `main.py` (approval endpoint), `tests/test_api.py` |
| 15 | `app/dashboard/app.py` |
| 16 | `scripts/replay_day.py` |

---

### Task 1: Project Setup

**Files:** `pyproject.toml`, `.env.example`, `.gitignore`, `config/settings.yaml`, `app/config.py`, `main.py`, `tests/conftest.py`

- [ ] **Step 1: Initialize git**

```bash
git init
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[project]
name = "ai-trading-bot"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "apscheduler>=3.10.4",
    "sqlalchemy>=2.0.0",
    "pydantic>=2.9.0",
    "pydantic-settings>=2.6.0",
    "anthropic>=0.40.0",
    "alpaca-py>=0.35.0",
    "finnhub-python>=2.4.20",
    "yfinance>=0.2.50",
    "pandas>=2.2.0",
    "pandas-ta>=0.3.14b",
    "pandas-market-calendars>=4.4.1",
    "streamlit>=1.40.0",
    "python-dotenv>=1.0.1",
    "pyyaml>=6.0.2",
    "httpx>=0.27.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-mock>=3.14.0",
    "freezegun>=1.5.0",
    "httpx>=0.27.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Create .env.example and .gitignore**

`.env.example`:
```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
FINNHUB_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

`.gitignore`:
```
.env
__pycache__/
*.pyc
.pytest_cache/
data/trading.db
data/test_*.db
reports/
.venv/
```

- [ ] **Step 4: Create config/settings.yaml**

```yaml
trading:
  mode: dry_run
  symbol: SPY
capital:
  starting: 100.0
  min_balance: 10.0
  max_position_pct: 0.20
risk:
  max_daily_trades: 1
  max_spread_pct: 0.0005
  risk_score_threshold: 0.65
  ai_confidence_threshold: 0.70
  max_drawdown_pct: 0.10
  consecutive_loss_limit: 3
  cooldown_minutes: 30
  earnings_proximity_days: 3
  fill_poll_timeout_seconds: 30
strategy:
  vix_max: 25.0
  rsi_max_long: 70.0
  relative_volume_min: 1.2
  atr_stop_multiplier: 2.0
  atr_target_multiplier: 3.0
versions:
  pipeline: "1.0.0"
  strategy: "1.0.0"
  prompt: "1.0.0"
dashboard:
  refresh_seconds: 60
  page_size: 50
```

- [ ] **Step 5: Create app/config.py**

```python
from pydantic_settings import BaseSettings
from typing import Literal
import yaml, os

class Settings(BaseSettings):
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    finnhub_api_key: str = ""
    anthropic_api_key: str = ""
    trading_mode: Literal["dry_run","paper_auto","live_manual","live_auto"] = "dry_run"
    trading_symbol: str = "SPY"
    starting_capital: float = 100.0
    min_balance_threshold: float = 10.0
    max_position_size_pct: float = 0.20
    max_daily_trades: int = 1
    max_spread_pct: float = 0.0005
    risk_score_threshold: float = 0.65
    ai_confidence_threshold: float = 0.70
    max_drawdown_pct: float = 0.10
    consecutive_loss_limit: int = 3
    cooldown_minutes: int = 30
    earnings_proximity_days: int = 3
    fill_poll_timeout_seconds: int = 30
    vix_max: float = 25.0
    rsi_max_long: float = 70.0
    relative_volume_min: float = 1.2
    atr_stop_multiplier: float = 2.0
    atr_target_multiplier: float = 3.0
    pipeline_version: str = "1.0.0"
    strategy_version: str = "1.0.0"
    prompt_version: str = "1.0.0"
    model_version: str = "claude-sonnet-4-6"
    dashboard_refresh_seconds: int = 60
    trade_history_page_size: int = 50
    database_url: str = "sqlite:///data/trading.db"

    class Config:
        env_file = ".env"

def _apply_yaml(path: str = "config/settings.yaml") -> None:
    if not os.path.exists(path):
        return
    with open(path) as f:
        d = yaml.safe_load(f)
    flat = {
        "TRADING_MODE": d["trading"]["mode"],
        "TRADING_SYMBOL": d["trading"]["symbol"],
        "STARTING_CAPITAL": d["capital"]["starting"],
        "MIN_BALANCE_THRESHOLD": d["capital"]["min_balance"],
        "MAX_POSITION_SIZE_PCT": d["capital"]["max_position_pct"],
        "MAX_DAILY_TRADES": d["risk"]["max_daily_trades"],
        "MAX_SPREAD_PCT": d["risk"]["max_spread_pct"],
        "RISK_SCORE_THRESHOLD": d["risk"]["risk_score_threshold"],
        "AI_CONFIDENCE_THRESHOLD": d["risk"]["ai_confidence_threshold"],
        "MAX_DRAWDOWN_PCT": d["risk"]["max_drawdown_pct"],
        "CONSECUTIVE_LOSS_LIMIT": d["risk"]["consecutive_loss_limit"],
        "COOLDOWN_MINUTES": d["risk"]["cooldown_minutes"],
        "EARNINGS_PROXIMITY_DAYS": d["risk"]["earnings_proximity_days"],
        "FILL_POLL_TIMEOUT_SECONDS": d["risk"]["fill_poll_timeout_seconds"],
        "VIX_MAX": d["strategy"]["vix_max"],
        "RSI_MAX_LONG": d["strategy"]["rsi_max_long"],
        "RELATIVE_VOLUME_MIN": d["strategy"]["relative_volume_min"],
        "ATR_STOP_MULTIPLIER": d["strategy"]["atr_stop_multiplier"],
        "ATR_TARGET_MULTIPLIER": d["strategy"]["atr_target_multiplier"],
        "PIPELINE_VERSION": d["versions"]["pipeline"],
        "STRATEGY_VERSION": d["versions"]["strategy"],
        "PROMPT_VERSION": d["versions"]["prompt"],
        "DASHBOARD_REFRESH_SECONDS": d["dashboard"]["refresh_seconds"],
        "TRADE_HISTORY_PAGE_SIZE": d["dashboard"]["page_size"],
    }
    for k, v in flat.items():
        os.environ.setdefault(k, str(v))

_apply_yaml()
settings = Settings()
```

- [ ] **Step 6: Create directory skeleton and main.py**

```bash
mkdir -p app/models app/pipeline app/data app/features app/strategies
mkdir -p app/ai app/prompts app/risk app/validation app/execution
mkdir -p app/persistence app/dashboard config data reports
mkdir -p tests/persistence tests/data tests/features tests/strategies
mkdir -p tests/ai tests/risk tests/validation tests/execution tests/pipeline
touch app/__init__.py app/models/__init__.py app/pipeline/__init__.py
touch app/data/__init__.py app/features/__init__.py app/strategies/__init__.py
touch app/ai/__init__.py app/risk/__init__.py app/validation/__init__.py
touch app/execution/__init__.py app/persistence/__init__.py app/dashboard/__init__.py
touch tests/__init__.py tests/persistence/__init__.py tests/data/__init__.py
touch tests/features/__init__.py tests/strategies/__init__.py tests/ai/__init__.py
touch tests/risk/__init__.py tests/validation/__init__.py
touch tests/execution/__init__.py tests/pipeline/__init__.py
```

`main.py`:
```python
from fastapi import FastAPI

app = FastAPI(title="AI Trading Bot", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 7: Create tests/conftest.py**

```python
import pytest
from datetime import datetime, date
from uuid import uuid4

@pytest.fixture
def cycle_id():
    return uuid4()

@pytest.fixture
def trade_date():
    return date(2026, 5, 16)
```

- [ ] **Step 8: Install deps and verify**

```bash
pip install -e ".[dev]"
python -c "import fastapi, sqlalchemy, anthropic; print('deps ok')"
pytest tests/ -v
```

Expected: `deps ok`, 0 tests collected (no failures).

- [ ] **Step 9: Commit**

```bash
git add .
git commit -m "feat: project setup, config, directory structure"
```

---

### Task 2: Pydantic Models

**Files:** `app/models/market.py`, `app/models/signals.py`, `app/models/ai.py`, `app/models/risk.py`, `app/models/execution.py`, `app/pipeline/context.py`, `app/pipeline/models.py`, `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

`tests/test_models.py`:
```python
from uuid import uuid4
from datetime import datetime
from app.models.market import MarketData, FeatureSet
from app.models.signals import TradeSignal, TradeRejection
from app.models.ai import AIAnalysis
from app.models.risk import RiskDecision, ValidationResult, KillSwitchState
from app.models.execution import OrderResult, FillRecord
from app.pipeline.context import PipelineContext

def test_feature_set_has_version():
    fs = FeatureSet(
        symbol="SPY", timeframe="1D", timestamp=datetime.utcnow(),
        feature_version="1.0.0", price=521.0, vwap=520.0,
        ema_9=521.5, ema_20=519.0, ema_50=515.0,
        rsi=55.0, macd=0.5, macd_signal=0.3, macd_hist=0.2,
        atr=2.1, vix=18.0, relative_volume=1.5, spread_proxy=0.0003,
        news_sentiment=0.2,
    )
    assert fs.feature_version == "1.0.0"

def test_trade_signal_type_literal():
    sig = TradeSignal(
        symbol="SPY", strategy="spy_trend_following",
        direction="long", strategy_confidence=0.74,
        entry_price=521.0, stop_loss=516.8, take_profit=527.3,
    )
    assert sig.type == "SIGNAL"

def test_trade_rejection_has_reasons():
    rej = TradeRejection(
        symbol="SPY", strategy="spy_trend_following",
        reasons=["VIX too high"], strategy_confidence=0.3,
    )
    assert rej.type == "REJECTION"
    assert len(rej.reasons) == 1

def test_ai_analysis_separates_confidence():
    a = AIAnalysis(
        decision="APPROVE", ai_confidence=0.81, regime="bullish",
        reasoning="ok", risk_factors=[], no_trade_reasons=[],
        raw_prompt="p", raw_response="r",
        model_version="claude-sonnet-4-6", prompt_version="1.0.0",
    )
    assert a.ai_confidence == 0.81
    assert not hasattr(a, "strategy_confidence")

def test_pipeline_context_starts_empty():
    ctx = PipelineContext()
    assert ctx.market_data is None
    assert ctx.signal is None
    assert ctx.errors == []
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/test_models.py -v
```

- [ ] **Step 3: Create app/models/market.py**

```python
from pydantic import BaseModel
from datetime import datetime

class MarketData(BaseModel):
    symbol: str
    timestamp: datetime
    pipeline_version: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float | None = None
    ask: float | None = None
    spread_proxy: float = 0.0
    vix: float = 20.0
    news_sentiment: float = 0.0
    bars_daily: list[dict] = []

class FeatureSet(BaseModel):
    symbol: str
    timeframe: str
    timestamp: datetime
    feature_version: str
    price: float
    vwap: float
    ema_9: float
    ema_20: float
    ema_50: float
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    atr: float
    vix: float
    relative_volume: float
    spread_proxy: float
    news_sentiment: float
```

- [ ] **Step 4: Create app/models/signals.py**

```python
from pydantic import BaseModel, Field
from typing import Literal
from uuid import UUID, uuid4

class TradeSignal(BaseModel):
    type: Literal["SIGNAL"] = "SIGNAL"
    trade_id: UUID = Field(default_factory=uuid4)
    symbol: str
    strategy: str
    direction: Literal["long", "short"]
    strategy_confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_side: Literal["long", "short"] = "long"

class TradeRejection(BaseModel):
    type: Literal["REJECTION"] = "REJECTION"
    symbol: str
    strategy: str
    reasons: list[str]
    strategy_confidence: float
```

- [ ] **Step 5: Create app/models/ai.py**

```python
from pydantic import BaseModel
from typing import Literal

class AIAnalysis(BaseModel):
    decision: Literal["APPROVE", "REJECT", "REDUCE_CONFIDENCE", "NO_TRADE"]
    ai_confidence: float
    regime: Literal["bullish", "bearish", "neutral", "uncertain"]
    reasoning: str
    risk_factors: list[str]
    no_trade_reasons: list[str]
    raw_prompt: str
    raw_response: str
    model_version: str
    prompt_version: str
    failed: bool = False
```

- [ ] **Step 6: Create app/models/risk.py**

```python
from pydantic import BaseModel
from typing import Literal
from datetime import datetime

class KillSwitchState(BaseModel):
    active: bool = False
    reason: str = ""
    activated_at: datetime | None = None

class RiskDecision(BaseModel):
    outcome: Literal["APPROVED", "BLOCKED", "KILL"]
    reason: str
    risk_score: float = 0.0
    tier: Literal["kill_switch", "hard_block", "score", "none"] = "none"

class ValidationResult(BaseModel):
    outcome: Literal["PASS", "FAIL"]
    reason: str
```

- [ ] **Step 7: Create app/models/execution.py**

```python
from pydantic import BaseModel
from typing import Literal
from datetime import datetime
from uuid import UUID

class OrderResult(BaseModel):
    execution_id: UUID
    trade_id: UUID
    broker_order_id: str
    symbol: str
    qty: float
    side: Literal["buy", "sell"]
    position_side: Literal["long", "short"]
    submitted_at: datetime
    broker_state: Literal["submitted","filled","partial_fill","canceled","reconciled","stale"] = "submitted"
    execution_fingerprint: str = ""

class FillRecord(BaseModel):
    execution_id: UUID
    fill_price: float
    fill_time: datetime
    slippage: float
    broker_state: Literal["filled","partial_fill","stale","reconciled"]
```

- [ ] **Step 8: Create app/pipeline/context.py**

```python
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional
from app.models.market import MarketData, FeatureSet
from app.models.signals import TradeSignal, TradeRejection
from app.models.ai import AIAnalysis
from app.models.risk import RiskDecision, ValidationResult
from app.models.execution import OrderResult

@dataclass
class PipelineContext:
    cycle_id: UUID = field(default_factory=uuid4)
    started_at: datetime = field(default_factory=datetime.utcnow)
    trading_mode: str = "dry_run"
    market_data: Optional[MarketData] = None
    features: Optional[FeatureSet] = None
    signal: Optional[TradeSignal | TradeRejection] = None
    ai_analysis: Optional[AIAnalysis] = None
    risk: Optional[RiskDecision] = None
    validation: Optional[ValidationResult] = None
    order: Optional[OrderResult] = None
    errors: list[str] = field(default_factory=list)
```

- [ ] **Step 9: Create app/pipeline/models.py**

```python
from typing import Literal
ReplayMode = Literal["full_pipeline", "feature_only", "strategy_only", "ai_only"]
```

- [ ] **Step 10: Run tests — expect PASS**

```bash
pytest tests/test_models.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 11: Commit**

```bash
git add app/models/ app/pipeline/context.py app/pipeline/models.py tests/test_models.py
git commit -m "feat: pydantic models for all pipeline stages"
```

---

### Task 3: Database Schema + DB Layer

**Files:** `app/persistence/db.py`, `scripts/seed_db.py`, `tests/persistence/test_db.py`

- [ ] **Step 1: Write failing tests**

`tests/persistence/test_db.py`:
```python
import pytest
from sqlalchemy import create_engine, inspect
from app.persistence.db import Base, get_session, CycleRecord, PendingTradeRecord, KillSwitchRecord

TEST_DB = "sqlite:///data/test_schema.db"

@pytest.fixture
def engine():
    e = create_engine(TEST_DB)
    Base.metadata.create_all(e)
    yield e
    Base.metadata.drop_all(e)

def test_required_tables_exist(engine):
    tables = inspect(engine).get_table_names()
    for t in ["cycles","trade_executions","positions","daily_summary",
               "pending_trades","kill_switch_state"]:
        assert t in tables, f"Missing table: {t}"

def test_kill_switch_singleton(engine):
    with get_session(engine) as s:
        s.add(KillSwitchRecord(id=1, active=False, reason=""))
        s.commit()
        rec = s.get(KillSwitchRecord, 1)
        assert rec.active is False

def test_pending_trade_status(engine):
    from uuid import uuid4
    from datetime import datetime
    with get_session(engine) as s:
        s.add(PendingTradeRecord(
            id=str(uuid4()), cycle_id=str(uuid4()),
            signal_json={}, status="PENDING_APPROVAL",
            created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
        ))
        s.commit()
        rec = s.query(PendingTradeRecord).first()
        assert rec.status == "PENDING_APPROVAL"
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/persistence/test_db.py -v
```

- [ ] **Step 3: Create app/persistence/db.py**

```python
from sqlalchemy import (Column, String, Float, Integer, DateTime, JSON, Boolean)
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from contextlib import contextmanager
from app.config import settings

class Base(DeclarativeBase):
    pass

class CycleRecord(Base):
    __tablename__ = "cycles"
    cycle_id = Column(String, primary_key=True)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    trading_mode = Column(String)
    pipeline_version = Column(String)
    strategy_version = Column(String)
    ai_prompt_version = Column(String)
    model_version = Column(String)
    market_data_json = Column(JSON)
    features_json = Column(JSON)
    signal_json = Column(JSON)
    ai_analysis_json = Column(JSON)
    risk_json = Column(JSON)
    validation_json = Column(JSON)
    errors_json = Column(JSON)

class TradeExecutionRecord(Base):
    __tablename__ = "trade_executions"
    execution_id = Column(String, primary_key=True)
    trade_id = Column(String, unique=True, nullable=False)
    cycle_id = Column(String)
    symbol = Column(String)
    direction = Column(String)
    entry_price = Column(Float)
    fill_price = Column(Float)
    slippage = Column(Float)
    qty = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    broker_state = Column(String)
    position_side = Column(String)
    execution_fingerprint = Column(String)
    pipeline_version = Column(String)
    strategy_version = Column(String)
    ai_prompt_version = Column(String)
    model_version = Column(String)
    created_at = Column(DateTime)
    filled_at = Column(DateTime)

class PositionRecord(Base):
    __tablename__ = "positions"
    id = Column(String, primary_key=True)
    trade_id = Column(String, nullable=False)
    symbol = Column(String)
    direction = Column(String)
    qty = Column(Float)
    entry_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    status = Column(String)
    pnl = Column(Float)
    opened_at = Column(DateTime)
    closed_at = Column(DateTime)

class DailySummaryRecord(Base):
    __tablename__ = "daily_summary"
    date = Column(String, primary_key=True)
    total_pnl = Column(Float)
    drawdown_pct = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    trade_count = Column(Integer)
    consecutive_losses = Column(Integer)
    strategy_performance = Column(JSON)
    pipeline_version = Column(String)

class PendingTradeRecord(Base):
    __tablename__ = "pending_trades"
    id = Column(String, primary_key=True)
    cycle_id = Column(String)
    signal_json = Column(JSON)
    status = Column(String)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

class KillSwitchRecord(Base):
    __tablename__ = "kill_switch_state"
    id = Column(Integer, primary_key=True, default=1)
    active = Column(Boolean, default=False)
    reason = Column(String, default="")
    activated_at = Column(DateTime)
    reset_at = Column(DateTime)

def make_engine(url: str | None = None):
    from sqlalchemy import create_engine
    return create_engine(url or settings.database_url)

@contextmanager
def get_session(engine=None):
    if engine is None:
        engine = make_engine()
    Session = sessionmaker(bind=engine)
    s = Session()
    try:
        yield s
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
```

- [ ] **Step 4: Create scripts/seed_db.py**

```python
import os
os.makedirs("data", exist_ok=True)
os.makedirs("reports", exist_ok=True)
from app.persistence.db import Base, KillSwitchRecord, make_engine, get_session
engine = make_engine()
Base.metadata.create_all(engine)
with get_session(engine) as s:
    if not s.get(KillSwitchRecord, 1):
        s.add(KillSwitchRecord(id=1, active=False, reason=""))
        s.commit()
print("Database initialized.")
```

- [ ] **Step 5: Run seed and tests**

```bash
python scripts/seed_db.py
pytest tests/persistence/test_db.py -v
```

Expected: `Database initialized.` + all 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add app/persistence/db.py scripts/seed_db.py tests/persistence/test_db.py
git commit -m "feat: SQLAlchemy schema, seed script, DB session helper"
```

---

### Task 4: Data Fetcher

**Files:** `app/data/fetcher.py`, `tests/data/test_fetcher.py`

- [ ] **Step 1: Write failing tests**

`tests/data/test_fetcher.py`:
```python
import pytest
from datetime import datetime
from app.data.fetcher import DataFetcher
from app.models.market import MarketData

@pytest.fixture
def fetcher():
    return DataFetcher(alpaca_api_key="test", alpaca_secret_key="test",
                       finnhub_api_key="test")

def _stub_bars(n=25):
    return [{"t": datetime.utcnow(), "o": 520.0, "h": 522.0,
             "l": 519.0, "c": 521.0, "v": 1_000_000.0}] * n

def test_fetch_returns_market_data(fetcher, mocker):
    mocker.patch.object(fetcher, "_fetch_bars_alpaca", return_value=_stub_bars())
    mocker.patch.object(fetcher, "_fetch_quote_alpaca", return_value=(521.0, 521.1))
    mocker.patch.object(fetcher, "_fetch_vix", return_value=18.5)
    mocker.patch.object(fetcher, "_fetch_news_sentiment", return_value=0.2)
    result = fetcher.fetch("SPY")
    assert isinstance(result, MarketData)
    assert result.symbol == "SPY"
    assert result.vix == 18.5

def test_spread_proxy_computed(fetcher, mocker):
    mocker.patch.object(fetcher, "_fetch_bars_alpaca", return_value=_stub_bars())
    mocker.patch.object(fetcher, "_fetch_quote_alpaca", return_value=(521.0, 521.2))
    mocker.patch.object(fetcher, "_fetch_vix", return_value=18.5)
    mocker.patch.object(fetcher, "_fetch_news_sentiment", return_value=0.0)
    result = fetcher.fetch("SPY")
    expected = (521.2 - 521.0) / ((521.0 + 521.2) / 2)
    assert abs(result.spread_proxy - expected) < 0.0001
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/data/test_fetcher.py -v
```

- [ ] **Step 3: Create app/data/fetcher.py**

```python
from datetime import datetime, timedelta
from app.models.market import MarketData
from app.config import settings

class DataFetcher:
    def __init__(self, alpaca_api_key: str, alpaca_secret_key: str, finnhub_api_key: str):
        from alpaca.data.historical import StockHistoricalDataClient
        import finnhub
        self._data_client = StockHistoricalDataClient(alpaca_api_key, alpaca_secret_key)
        self._finnhub = finnhub.Client(api_key=finnhub_api_key)

    def fetch(self, symbol: str) -> MarketData:
        bars = self._fetch_bars_alpaca(symbol)
        bid, ask = self._fetch_quote_alpaca(symbol)
        vix = self._fetch_vix()
        sentiment = self._fetch_news_sentiment(symbol)
        mid = (bid + ask) / 2
        spread_proxy = (ask - bid) / mid if mid > 0 else 0.0
        latest = bars[-1] if bars else {}
        return MarketData(
            symbol=symbol, timestamp=datetime.utcnow(),
            pipeline_version=settings.pipeline_version,
            open=latest.get("o", 0.0), high=latest.get("h", 0.0),
            low=latest.get("l", 0.0), close=latest.get("c", 0.0),
            volume=latest.get("v", 0.0),
            bid=bid, ask=ask, spread_proxy=spread_proxy,
            bars_daily=bars, vix=vix, news_sentiment=sentiment,
        )

    def _fetch_bars_alpaca(self, symbol: str) -> list[dict]:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        req = StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
            start=datetime.utcnow() - timedelta(days=30),
            end=datetime.utcnow(),
        )
        bars = self._data_client.get_stock_bars(req)
        df = bars.df
        if df.empty:
            return []
        return [{"t": idx[1] if isinstance(idx, tuple) else idx,
                 "o": float(row["open"]), "h": float(row["high"]),
                 "l": float(row["low"]), "c": float(row["close"]),
                 "v": float(row["volume"])}
                for idx, row in df.iterrows()]

    def _fetch_quote_alpaca(self, symbol: str) -> tuple[float, float]:
        from alpaca.data.requests import StockLatestQuoteRequest
        q = self._data_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=symbol))[symbol]
        return float(q.bid_price), float(q.ask_price)

    def _fetch_vix(self) -> float:
        try:
            import yfinance as yf
            return float(yf.Ticker("^VIX").fast_info["last_price"])
        except Exception:
            return 20.0

    def _fetch_news_sentiment(self, symbol: str) -> float:
        try:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            news = self._finnhub.company_news(symbol, _from=today, to=today)
            scores = [n["sentiment"]["score"] for n in (news or [])[:10]
                      if "sentiment" in n]
            return sum(scores) / len(scores) if scores else 0.0
        except Exception:
            return 0.0

def make_fetcher() -> DataFetcher:
    return DataFetcher(
        alpaca_api_key=settings.alpaca_api_key,
        alpaca_secret_key=settings.alpaca_secret_key,
        finnhub_api_key=settings.finnhub_api_key,
    )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/data/test_fetcher.py -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add app/data/fetcher.py tests/data/test_fetcher.py
git commit -m "feat: data fetcher (Alpaca OHLCV, VIX via yfinance, Finnhub news)"
```

---

### Task 5: Feature Pipeline

**Files:** `app/features/pipeline.py`, `tests/features/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

`tests/features/test_pipeline.py`:
```python
import pytest
from datetime import datetime, timedelta
from app.features.pipeline import FeaturePipeline
from app.models.market import MarketData, FeatureSet

def make_market_data(n=30):
    bars = [{"t": datetime.utcnow() - timedelta(days=n-i),
             "o": 520.0+i*0.1, "h": 521.0+i*0.1, "l": 519.0+i*0.1,
             "c": 520.5+i*0.1, "v": 1_000_000+i*10_000}
            for i in range(n)]
    return MarketData(
        symbol="SPY", timestamp=datetime.utcnow(), pipeline_version="1.0.0",
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
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/features/test_pipeline.py -v
```

- [ ] **Step 3: Create app/features/pipeline.py**

```python
import pandas as pd
import pandas_ta as ta
from datetime import datetime
from app.models.market import MarketData, FeatureSet
from app.config import settings

class FeaturePipeline:
    @staticmethod
    def compute(market_data: MarketData) -> FeatureSet:
        df = FeaturePipeline._to_df(market_data.bars_daily)
        close = df["close"]

        ema_9  = float(ta.ema(close, length=9).iloc[-1])
        ema_20 = float(ta.ema(close, length=20).iloc[-1])
        ema_50 = float(ta.ema(close, length=50).iloc[-1])
        rsi    = float(ta.rsi(close, length=14).iloc[-1])

        macd_df     = ta.macd(close, fast=12, slow=26, signal=9)
        macd        = float(macd_df["MACD_12_26_9"].iloc[-1])
        macd_signal = float(macd_df["MACDs_12_26_9"].iloc[-1])
        macd_hist   = float(macd_df["MACDh_12_26_9"].iloc[-1])

        atr = float(ta.atr(df["high"], df["low"], close, length=14).iloc[-1])

        avg_vol = float(df["volume"].iloc[-21:-1].mean())
        rel_vol = float(df["volume"].iloc[-1]) / avg_vol if avg_vol > 0 else 1.0

        typical = (df["high"] + df["low"] + close) / 3
        vwap = float((typical * df["volume"]).sum() / df["volume"].sum())

        return FeatureSet(
            symbol=market_data.symbol,
            timeframe="1D",
            timestamp=market_data.timestamp,
            feature_version=settings.pipeline_version,
            price=market_data.close,
            vwap=vwap,
            ema_9=ema_9, ema_20=ema_20, ema_50=ema_50,
            rsi=rsi, macd=macd, macd_signal=macd_signal, macd_hist=macd_hist,
            atr=atr,
            vix=market_data.vix,
            relative_volume=rel_vol,
            spread_proxy=market_data.spread_proxy,
            news_sentiment=market_data.news_sentiment,
        )

    @staticmethod
    def _to_df(bars: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(bars).rename(columns={
            "t": "timestamp", "o": "open", "h": "high",
            "l": "low", "c": "close", "v": "volume",
        })
        return df.sort_values("timestamp").reset_index(drop=True)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/features/test_pipeline.py -v
```

Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add app/features/pipeline.py tests/features/test_pipeline.py
git commit -m "feat: feature pipeline (EMA, RSI, MACD, ATR, rel vol, VWAP)"
```

---

### Task 6: SPY Trend Strategy

**Files:** `app/strategies/spy_trend.py`, `tests/strategies/test_spy_trend.py`

- [ ] **Step 1: Write failing tests**

`tests/strategies/test_spy_trend.py`:
```python
import pytest
from datetime import datetime
from app.strategies.spy_trend import SpyTrendStrategy
from app.models.market import FeatureSet
from app.models.signals import TradeSignal, TradeRejection

def make_fs(**ov):
    d = dict(symbol="SPY", timeframe="1D", timestamp=datetime.utcnow(),
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
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/strategies/test_spy_trend.py -v
```

- [ ] **Step 3: Create app/strategies/spy_trend.py**

```python
from app.models.market import FeatureSet
from app.models.signals import TradeSignal, TradeRejection
from app.config import settings

class SpyTrendStrategy:
    NAME = "spy_trend_following"

    @staticmethod
    def evaluate(f: FeatureSet) -> TradeSignal | TradeRejection:
        failed = []
        if f.price <= f.ema_20:
            failed.append("Price below EMA20")
        if f.ema_9 <= f.ema_20:
            failed.append("EMA9 not above EMA20")
        if f.relative_volume < settings.relative_volume_min:
            failed.append(f"Relative volume {f.relative_volume:.2f} below {settings.relative_volume_min}")
        if f.vix >= settings.vix_max:
            failed.append(f"VIX {f.vix:.1f} >= max {settings.vix_max}")
        if f.rsi >= settings.rsi_max_long:
            failed.append(f"RSI {f.rsi:.1f} overbought")

        conf = SpyTrendStrategy._confidence(f)

        if failed:
            return TradeRejection(symbol=f.symbol, strategy=SpyTrendStrategy.NAME,
                                  reasons=failed, strategy_confidence=conf)

        entry = f.price
        return TradeSignal(
            symbol=f.symbol, strategy=SpyTrendStrategy.NAME, direction="long",
            strategy_confidence=conf, entry_price=entry,
            stop_loss=entry - f.atr * settings.atr_stop_multiplier,
            take_profit=entry + f.atr * settings.atr_target_multiplier,
            position_side="long",
        )

    @staticmethod
    def _confidence(f: FeatureSet) -> float:
        ema_gap    = max(0.0, (f.ema_9 - f.ema_20) / f.ema_20) * 100
        ema_score  = min(1.0, ema_gap / 2.0)
        vol_score  = min(1.0, max(0.0, (f.relative_volume - 1.0) / 2.0))
        price_gap  = max(0.0, (f.price - f.ema_20) / f.ema_20) * 100
        trend_score= min(1.0, price_gap / 2.0)
        vix_score  = max(0.0, 1.0 - f.vix / settings.vix_max)
        return round(ema_score*0.30 + vol_score*0.20 + trend_score*0.30 + vix_score*0.20, 4)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/strategies/test_spy_trend.py -v
```

Expected: all 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add app/strategies/spy_trend.py tests/strategies/test_spy_trend.py
git commit -m "feat: SPY trend strategy, TradeSignal|TradeRejection, monotonic confidence"
```

---

### Task 7: AI Analyst

**Files:** `app/prompts/system_prompt.txt`, `app/prompts/market_analysis.txt`, `app/ai/analyst.py`, `tests/ai/test_analyst.py`

- [ ] **Step 1: Write failing tests**

`tests/ai/test_analyst.py`:
```python
import json
from unittest.mock import MagicMock
from datetime import datetime
from uuid import uuid4
from app.ai.analyst import AIAnalyst
from app.models.signals import TradeSignal, TradeRejection
from app.models.ai import AIAnalysis
from app.models.market import FeatureSet

def make_fs():
    return FeatureSet(symbol="SPY", timeframe="1D", timestamp=datetime.utcnow(),
                      feature_version="1.0.0", price=521.0, vwap=519.0,
                      ema_9=521.5, ema_20=519.0, ema_50=515.0, rsi=55.0,
                      macd=0.5, macd_signal=0.3, macd_hist=0.2, atr=2.1,
                      vix=18.0, relative_volume=1.5, spread_proxy=0.0003,
                      news_sentiment=0.1)

def make_signal():
    return TradeSignal(trade_id=uuid4(), symbol="SPY",
                       strategy="spy_trend_following", direction="long",
                       strategy_confidence=0.74, entry_price=521.0,
                       stop_loss=516.8, take_profit=527.3)

def mock_analyst(response_dict: dict) -> AIAnalyst:
    a = AIAnalyst.__new__(AIAnalyst)
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps(response_dict))]
    client = MagicMock()
    client.messages.create.return_value = msg
    a._client = client
    return a

def test_rejection_skips_llm():
    a = AIAnalyst.__new__(AIAnalyst)
    a._client = MagicMock()
    rej = TradeRejection(symbol="SPY", strategy="spy_trend_following",
                         reasons=["VIX too high"], strategy_confidence=0.3)
    result = a.analyze(rej, make_fs())
    a._client.messages.create.assert_not_called()
    assert result.decision == "REJECT"

def test_approve_parses_correctly():
    a = mock_analyst({"decision":"APPROVE","ai_confidence":0.82,"regime":"bullish",
                      "reasoning":"ok","risk_factors":[],"no_trade_reasons":[]})
    r = a.analyze(make_signal(), make_fs())
    assert isinstance(r, AIAnalysis)
    assert r.decision == "APPROVE"
    assert r.ai_confidence == 0.82
    assert r.failed is False

def test_parse_failure_returns_failed_analysis():
    a = AIAnalyst.__new__(AIAnalyst)
    msg = MagicMock()
    msg.content = [MagicMock(text="not json {{")]
    a._client = MagicMock()
    a._client.messages.create.return_value = msg
    r = a.analyze(make_signal(), make_fs())
    assert r.failed is True
    assert r.decision == "NO_TRADE"

def test_raw_fields_stored():
    a = mock_analyst({"decision":"APPROVE","ai_confidence":0.75,"regime":"neutral",
                      "reasoning":"ok","risk_factors":[],"no_trade_reasons":[]})
    r = a.analyze(make_signal(), make_fs())
    assert len(r.raw_prompt) > 0
    assert len(r.raw_response) > 0
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/ai/test_analyst.py -v
```

- [ ] **Step 3: Create app/prompts/system_prompt.txt**

```
You are a conservative trading analyst for a small paper trading account.

STRATEGY AUTHORITY:
- The strategy engine is deterministic and must NOT be overridden.
- You are an annotator and risk evaluator only.
- If the signal type is REJECTION, you MUST output decision=REJECT.
- You do not generate trade ideas. You evaluate what the strategy produced.

RULES (non-negotiable, enforced in code — do not attempt to override):
- If ai_confidence < 0.70, set decision = NO_TRADE
- Stop loss is mandatory on every trade
- Max 1 trade per day
- Avoid elevated volatility (VIX > 25)

OUTPUT FORMAT — return only valid JSON matching this exact schema:
{
  "decision": "APPROVE" | "REJECT" | "REDUCE_CONFIDENCE" | "NO_TRADE",
  "ai_confidence": <float 0.0-1.0>,
  "regime": "bullish" | "bearish" | "neutral" | "uncertain",
  "reasoning": "<plain english explanation under 200 words>",
  "risk_factors": ["<risk>", ...],
  "no_trade_reasons": ["<reason if NO_TRADE or REJECT>"]
}
```

- [ ] **Step 4: Create app/prompts/market_analysis.txt**

```
TRADING MODE: {trading_mode}
ACCOUNT BALANCE: ${account_balance:.2f} (paper)

MARKET DATA:
{feature_set_json}

STRATEGY SIGNAL:
{trade_signal_json}

TASK:
1. Determine market regime
2. Evaluate signal quality
3. Identify risk factors
4. Set ai_confidence (0.0-1.0)
5. Output decision: APPROVE, REJECT, REDUCE_CONFIDENCE, or NO_TRADE

Rules reminder:
- If ai_confidence < 0.70 → NO_TRADE
- If signal type = REJECTION → REJECT immediately

Return only the JSON object described in your system prompt.
```

- [ ] **Step 5: Create app/ai/analyst.py**

```python
import json, os
import anthropic
from app.models.signals import TradeSignal, TradeRejection
from app.models.market import FeatureSet
from app.models.ai import AIAnalysis
from app.config import settings

_PROMPTS = os.path.join(os.path.dirname(__file__), "..", "prompts")

def _load(name: str) -> str:
    with open(os.path.join(_PROMPTS, name)) as f:
        return f.read()

class AIAnalyst:
    def __init__(self):
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def analyze(self, signal: TradeSignal | TradeRejection,
                features: FeatureSet, account_balance: float = 100.0) -> AIAnalysis:
        if isinstance(signal, TradeRejection):
            return AIAnalysis(
                decision="REJECT", ai_confidence=0.0, regime="uncertain",
                reasoning=f"Strategy rejected: {', '.join(signal.reasons)}",
                risk_factors=signal.reasons, no_trade_reasons=signal.reasons,
                raw_prompt="", raw_response="",
                model_version=settings.model_version, prompt_version=settings.prompt_version,
            )

        system = _load("system_prompt.txt")
        user = _load("market_analysis.txt").format(
            trading_mode=settings.trading_mode,
            account_balance=account_balance,
            feature_set_json=features.model_dump_json(indent=2),
            trade_signal_json=signal.model_dump_json(indent=2),
        )
        try:
            resp = self._client.messages.create(
                model=settings.model_version, max_tokens=1024,
                system=system, messages=[{"role": "user", "content": user}],
            )
            raw = resp.content[0].text
            data = json.loads(raw)
            return AIAnalysis(
                decision=data["decision"],
                ai_confidence=float(data["ai_confidence"]),
                regime=data["regime"],
                reasoning=data.get("reasoning", ""),
                risk_factors=data.get("risk_factors", []),
                no_trade_reasons=data.get("no_trade_reasons", []),
                raw_prompt=user, raw_response=raw,
                model_version=settings.model_version, prompt_version=settings.prompt_version,
            )
        except Exception as e:
            return AIAnalysis(
                decision="NO_TRADE", ai_confidence=0.0, regime="uncertain",
                reasoning=f"Failure: {e}", risk_factors=[], no_trade_reasons=[str(e)],
                raw_prompt=user if "user" in dir() else "",
                raw_response="",
                model_version=settings.model_version, prompt_version=settings.prompt_version,
                failed=True,
            )
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/ai/test_analyst.py -v
```

Expected: all 4 PASS.

- [ ] **Step 7: Commit**

```bash
git add app/ai/ app/prompts/ tests/ai/
git commit -m "feat: Claude AI analyst, rejection bypass, parse-failure isolation"
```

---

### Task 8: Risk Engine

**Files:** `app/risk/engine.py`, `tests/risk/test_engine.py`

- [ ] **Step 1: Write failing tests**

`tests/risk/test_engine.py`:
```python
import pytest
from uuid import uuid4
from datetime import datetime
from app.risk.engine import RiskEngine
from app.models.signals import TradeSignal
from app.models.ai import AIAnalysis
from app.models.risk import RiskDecision, KillSwitchState

def make_signal(conf=0.74):
    return TradeSignal(trade_id=uuid4(), symbol="SPY",
                       strategy="spy_trend_following", direction="long",
                       strategy_confidence=conf, entry_price=521.0,
                       stop_loss=516.8, take_profit=527.3)

def make_analysis(conf=0.82, decision="APPROVE"):
    return AIAnalysis(decision=decision, ai_confidence=conf, regime="bullish",
                      reasoning="ok", risk_factors=[], no_trade_reasons=[],
                      raw_prompt="", raw_response="",
                      model_version="claude-sonnet-4-6", prompt_version="1.0.0")

def stub_engine(balance=100.0, daily_trades=0, drawdown=0.0,
                consec=0, kill=False, last_trade=None):
    e = RiskEngine.__new__(RiskEngine)
    e._get_account_balance    = lambda: balance
    e._get_daily_trade_count  = lambda: daily_trades
    e._get_daily_drawdown     = lambda: drawdown
    e._get_consecutive_losses = lambda: consec
    e._get_kill_switch        = lambda: KillSwitchState(active=kill)
    e._get_last_trade_time    = lambda: last_trade
    e._is_earnings_proximity  = lambda sym: False
    e._activate_kill_switch   = lambda r: None
    return e

def test_approved_when_clear():
    assert stub_engine().evaluate(make_signal(), make_analysis(), vix=18.0).outcome == "APPROVED"

def test_kill_switch_active():
    r = stub_engine(kill=True).evaluate(make_signal(), make_analysis(), vix=18.0)
    assert r.outcome == "KILL" and r.tier == "kill_switch"

def test_drawdown_triggers_kill():
    assert stub_engine(drawdown=0.12).evaluate(make_signal(), make_analysis(), vix=18.0).outcome == "KILL"

def test_consec_losses_triggers_kill():
    assert stub_engine(consec=3).evaluate(make_signal(), make_analysis(), vix=18.0).outcome == "KILL"

def test_daily_limit_blocks():
    r = stub_engine(daily_trades=1).evaluate(make_signal(), make_analysis(), vix=18.0)
    assert r.outcome == "BLOCKED" and r.tier == "hard_block"

def test_low_balance_blocks():
    assert stub_engine(balance=5.0).evaluate(make_signal(), make_analysis(), vix=18.0).outcome == "BLOCKED"

def test_low_risk_score_blocks():
    r = stub_engine().evaluate(make_signal(conf=0.3), make_analysis(conf=0.3), vix=24.0)
    assert r.outcome == "BLOCKED" and r.tier == "score"

def test_risk_score_in_result():
    r = stub_engine().evaluate(make_signal(), make_analysis(), vix=18.0)
    assert 0.0 <= r.risk_score <= 1.0
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/risk/test_engine.py -v
```

- [ ] **Step 3: Create app/risk/engine.py**

```python
from datetime import datetime
from app.models.signals import TradeSignal
from app.models.ai import AIAnalysis
from app.models.risk import RiskDecision, KillSwitchState
from app.config import settings

class RiskEngine:
    def __init__(self, session_factory=None):
        self._sf = session_factory

    def evaluate(self, signal: TradeSignal, analysis: AIAnalysis,
                 vix: float = 20.0) -> RiskDecision:
        # Tier 1: Kill Switches
        ks = self._get_kill_switch()
        if ks.active:
            return RiskDecision(outcome="KILL", reason=f"Kill active: {ks.reason}",
                                tier="kill_switch")
        if self._get_daily_drawdown() >= settings.max_drawdown_pct:
            self._activate_kill_switch("Drawdown limit")
            return RiskDecision(outcome="KILL", reason="Daily drawdown >= 10%", tier="kill_switch")
        if self._get_consecutive_losses() >= settings.consecutive_loss_limit:
            self._activate_kill_switch("Consecutive losses")
            return RiskDecision(outcome="KILL", reason="3 consecutive losses", tier="kill_switch")

        # Tier 2: Hard Blocks
        balance = self._get_account_balance()
        if balance < settings.min_balance_threshold:
            return RiskDecision(outcome="BLOCKED", reason=f"Balance ${balance:.2f} below floor",
                                tier="hard_block")
        if self._get_daily_trade_count() >= settings.max_daily_trades:
            return RiskDecision(outcome="BLOCKED", reason="Daily trade limit", tier="hard_block")
        if self._is_earnings_proximity(signal.symbol):
            return RiskDecision(outcome="BLOCKED", reason="Earnings proximity", tier="hard_block")
        last = self._get_last_trade_time()
        if last:
            elapsed = (datetime.utcnow() - last).total_seconds() / 60
            if elapsed < settings.cooldown_minutes:
                return RiskDecision(outcome="BLOCKED",
                                    reason=f"Cooldown {settings.cooldown_minutes-elapsed:.0f}m",
                                    tier="hard_block")

        # Tier 3: Composite Score
        vix_score = max(0.0, 1.0 - vix / settings.vix_max)
        score = round(signal.strategy_confidence*0.40 +
                      analysis.ai_confidence*0.30 +
                      vix_score*0.30, 4)
        if score < settings.risk_score_threshold:
            return RiskDecision(outcome="BLOCKED",
                                reason=f"risk_score {score:.3f} < {settings.risk_score_threshold}",
                                risk_score=score, tier="score")

        return RiskDecision(outcome="APPROVED", reason="All checks passed",
                            risk_score=score, tier="none")

    def _get_account_balance(self) -> float:
        from alpaca.trading.client import TradingClient
        return float(TradingClient(settings.alpaca_api_key, settings.alpaca_secret_key,
                                   paper=True).get_account().cash)

    def _get_daily_trade_count(self) -> int:
        if not self._sf: return 0
        from app.persistence.db import TradeExecutionRecord
        from datetime import date
        with self._sf() as s:
            return s.query(TradeExecutionRecord).filter(
                TradeExecutionRecord.created_at >= date.today().isoformat(),
                TradeExecutionRecord.broker_state == "filled").count()

    def _get_daily_drawdown(self) -> float:
        if not self._sf: return 0.0
        from app.persistence.db import DailySummaryRecord
        from datetime import date
        with self._sf() as s:
            r = s.get(DailySummaryRecord, date.today().isoformat())
            return r.drawdown_pct if r else 0.0

    def _get_consecutive_losses(self) -> int:
        if not self._sf: return 0
        from app.persistence.db import DailySummaryRecord
        from datetime import date
        with self._sf() as s:
            r = s.get(DailySummaryRecord, date.today().isoformat())
            return r.consecutive_losses if r else 0

    def _get_kill_switch(self) -> KillSwitchState:
        if not self._sf: return KillSwitchState()
        from app.persistence.db import KillSwitchRecord
        with self._sf() as s:
            ks = s.get(KillSwitchRecord, 1)
            return KillSwitchState(active=ks.active if ks else False,
                                   reason=ks.reason or "" if ks else "")

    def _activate_kill_switch(self, reason: str) -> None:
        if not self._sf: return
        from app.persistence.db import KillSwitchRecord
        with self._sf() as s:
            ks = s.get(KillSwitchRecord, 1)
            if ks:
                ks.active, ks.reason, ks.activated_at = True, reason, datetime.utcnow()
                s.commit()

    def _get_last_trade_time(self) -> datetime | None:
        if not self._sf: return None
        from app.persistence.db import TradeExecutionRecord
        with self._sf() as s:
            r = s.query(TradeExecutionRecord).order_by(
                TradeExecutionRecord.created_at.desc()).first()
            return r.created_at if r else None

    def _is_earnings_proximity(self, symbol: str) -> bool:
        return False
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/risk/test_engine.py -v
```

Expected: all 8 PASS.

- [ ] **Step 5: Commit**

```bash
git add app/risk/engine.py tests/risk/test_engine.py
git commit -m "feat: risk engine, ordered kill switches, composite score"
```

---

### Task 9: Trade Validator

**Files:** `app/validation/validator.py`, `tests/validation/test_validator.py`

- [ ] **Step 1: Write failing tests**

`tests/validation/test_validator.py`:
```python
import pytest
from uuid import uuid4
from datetime import datetime
from app.validation.validator import TradeValidator
from app.models.signals import TradeSignal
from app.models.risk import ValidationResult

def make_signal():
    return TradeSignal(trade_id=uuid4(), symbol="SPY",
                       strategy="spy_trend_following", direction="long",
                       strategy_confidence=0.74, entry_price=521.0,
                       stop_loss=516.8, take_profit=527.3)

def stub(mode="paper_auto", spread=0.0002, power=100.0, open_pos=False):
    v = TradeValidator.__new__(TradeValidator)
    v._trading_mode    = mode
    v._get_spread      = lambda sym: spread
    v._get_buying_power= lambda: power
    v._has_open_position= lambda sym: open_pos
    v._write_pending   = lambda sig: None
    return v

def test_pass_when_clear():
    assert stub().validate(make_signal()).outcome == "PASS"

def test_fail_dry_run():
    r = stub(mode="dry_run").validate(make_signal())
    assert r.outcome == "FAIL" and "dry_run" in r.reason

def test_fail_wide_spread():
    r = stub(spread=0.01).validate(make_signal())
    assert r.outcome == "FAIL" and "spread" in r.reason.lower()

def test_fail_low_buying_power():
    assert stub(power=1.0).validate(make_signal()).outcome == "FAIL"

def test_fail_open_position():
    assert stub(open_pos=True).validate(make_signal()).outcome == "FAIL"

def test_live_manual_writes_pending():
    wrote = []
    v = stub(mode="live_manual")
    v._write_pending = lambda sig: wrote.append(sig)
    r = v.validate(make_signal())
    assert r.outcome == "FAIL" and "approval" in r.reason.lower()
    assert len(wrote) == 1
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/validation/test_validator.py -v
```

- [ ] **Step 3: Create app/validation/validator.py**

```python
from app.models.signals import TradeSignal
from app.models.risk import ValidationResult
from app.config import settings

class TradeValidator:
    def __init__(self, session_factory=None):
        self._sf = session_factory
        self._trading_mode = settings.trading_mode

    def validate(self, signal: TradeSignal) -> ValidationResult:
        if self._trading_mode == "dry_run":
            return ValidationResult(outcome="FAIL", reason="dry_run mode — no execution")

        spread = self._get_spread(signal.symbol)
        if spread > settings.max_spread_pct:
            return ValidationResult(outcome="FAIL",
                                    reason=f"spread {spread:.4%} > max {settings.max_spread_pct:.4%}")

        power = self._get_buying_power()
        if power < 10.0:
            return ValidationResult(outcome="FAIL", reason=f"buying power ${power:.2f} insufficient")

        if self._has_open_position(signal.symbol):
            return ValidationResult(outcome="FAIL", reason=f"open position in {signal.symbol}")

        if self._trading_mode == "live_manual":
            self._write_pending(signal)
            return ValidationResult(outcome="FAIL", reason="awaiting manual approval")

        return ValidationResult(outcome="PASS", reason="all checks passed")

    def _get_spread(self, symbol: str) -> float:
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest
            client = StockHistoricalDataClient(settings.alpaca_api_key, settings.alpaca_secret_key)
            q = client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))[symbol]
            mid = (q.bid_price + q.ask_price) / 2
            return (q.ask_price - q.bid_price) / mid if mid > 0 else 1.0
        except Exception:
            return 0.0

    def _get_buying_power(self) -> float:
        try:
            from alpaca.trading.client import TradingClient
            return float(TradingClient(settings.alpaca_api_key, settings.alpaca_secret_key,
                                       paper=True).get_account().buying_power)
        except Exception:
            return 0.0

    def _has_open_position(self, symbol: str) -> bool:
        try:
            from alpaca.trading.client import TradingClient
            positions = TradingClient(settings.alpaca_api_key, settings.alpaca_secret_key,
                                      paper=True).get_all_positions()
            return any(p.symbol == symbol for p in positions)
        except Exception:
            return False

    def _write_pending(self, signal: TradeSignal) -> None:
        if not self._sf: return
        from app.persistence.db import PendingTradeRecord
        from datetime import datetime
        with self._sf() as s:
            s.add(PendingTradeRecord(
                id=str(signal.trade_id), cycle_id="",
                signal_json=signal.model_dump(mode="json"),
                status="PENDING_APPROVAL",
                created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
            ))
            s.commit()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/validation/test_validator.py -v
```

Expected: all 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add app/validation/validator.py tests/validation/test_validator.py
git commit -m "feat: trade validator (spread, buying power, position, mode gate)"
```

---

### Task 10: Execution Engine

**Files:** `app/execution/executor.py`, `tests/execution/test_executor.py`

- [ ] **Step 1: Write failing tests**

`tests/execution/test_executor.py`:
```python
import pytest
from uuid import uuid4
from datetime import datetime
from unittest.mock import MagicMock
from app.execution.executor import ExecutionEngine
from app.models.signals import TradeSignal
from app.models.execution import OrderResult, FillRecord

def make_signal():
    return TradeSignal(trade_id=uuid4(), symbol="SPY",
                       strategy="spy_trend_following", direction="long",
                       strategy_confidence=0.74, entry_price=521.0,
                       stop_loss=516.8, take_profit=527.3)

def stub_engine(mode="paper_auto"):
    e = ExecutionEngine.__new__(ExecutionEngine)
    e._trading_mode = mode
    mock_order = MagicMock(); mock_order.id = "order-123"
    e._trading_client = MagicMock()
    e._trading_client.submit_order.return_value = mock_order
    e._is_already_executed = lambda tid: False
    e._persist_execution = lambda *a, **kw: None
    return e

def test_dry_run_returns_none():
    e = stub_engine(mode="dry_run")
    assert e.execute(make_signal(), account_balance=100.0) is None
    e._trading_client.submit_order.assert_not_called()

def test_paper_auto_submits(mocker):
    e = stub_engine()
    fill = FillRecord(execution_id=uuid4(), fill_price=521.05,
                      fill_time=datetime.utcnow(), slippage=0.05, broker_state="filled")
    mocker.patch.object(e, "_poll_fill", return_value=fill)
    result = e.execute(make_signal(), account_balance=100.0)
    e._trading_client.submit_order.assert_called_once()
    assert result is not None

def test_idempotency_blocks_duplicate(mocker):
    e = stub_engine()
    e._is_already_executed = lambda tid: True
    result = e.execute(make_signal(), account_balance=100.0)
    assert result is None
    e._trading_client.submit_order.assert_not_called()

def test_stale_fill_on_timeout(mocker):
    e = stub_engine()
    mocker.patch.object(e, "_poll_fill", return_value=None)
    result = e.execute(make_signal(), account_balance=100.0)
    assert result is not None
    assert result.broker_state == "stale"
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/execution/test_executor.py -v
```

- [ ] **Step 3: Create app/execution/executor.py**

```python
import time
from uuid import uuid4, UUID
from datetime import datetime
from app.models.signals import TradeSignal
from app.models.execution import OrderResult, FillRecord
from app.config import settings

class ExecutionEngine:
    def __init__(self, session_factory=None):
        self._sf = session_factory
        self._trading_mode = settings.trading_mode
        from alpaca.trading.client import TradingClient
        self._trading_client = TradingClient(
            settings.alpaca_api_key, settings.alpaca_secret_key, paper=True)

    def execute(self, signal: TradeSignal, account_balance: float,
                fingerprint: str = "") -> OrderResult | None:
        if self._trading_mode == "dry_run":
            return None
        if self._is_already_executed(signal.trade_id):
            return None

        qty = max(1.0, round((account_balance * settings.max_position_size_pct) / signal.entry_price))
        execution_id = uuid4()

        try:
            order = self._submit_bracket(signal, qty)
        except Exception:
            return None

        result = OrderResult(
            execution_id=execution_id, trade_id=signal.trade_id,
            broker_order_id=str(order.id), symbol=signal.symbol,
            qty=qty, side="buy", position_side=signal.position_side,
            submitted_at=datetime.utcnow(), execution_fingerprint=fingerprint,
        )

        fill = self._poll_fill(str(order.id), signal.entry_price)
        result.broker_state = fill.broker_state if fill else "stale"

        self._persist_execution(result, fill, signal)
        return result

    def _submit_bracket(self, signal: TradeSignal, qty: float):
        from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
        return self._trading_client.submit_order(MarketOrderRequest(
            symbol=signal.symbol, qty=qty, side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY, order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(signal.take_profit, 2)),
            stop_loss=StopLossRequest(stop_price=round(signal.stop_loss, 2)),
        ))

    def _poll_fill(self, order_id: str, entry_price: float) -> FillRecord | None:
        deadline = time.time() + settings.fill_poll_timeout_seconds
        while time.time() < deadline:
            try:
                o = self._trading_client.get_order_by_id(order_id)
                if o.status.value == "filled":
                    fp = float(o.filled_avg_price or entry_price)
                    return FillRecord(execution_id=uuid4(), fill_price=fp,
                                      fill_time=datetime.utcnow(),
                                      slippage=fp - entry_price, broker_state="filled")
            except Exception:
                pass
            time.sleep(2)
        return None

    def _is_already_executed(self, trade_id: UUID) -> bool:
        if not self._sf: return False
        from app.persistence.db import TradeExecutionRecord
        with self._sf() as s:
            return s.query(TradeExecutionRecord).filter_by(
                trade_id=str(trade_id)).first() is not None

    def _persist_execution(self, result: OrderResult, fill: FillRecord | None,
                           signal: TradeSignal) -> None:
        if not self._sf: return
        from app.persistence.db import TradeExecutionRecord
        with self._sf() as s:
            s.add(TradeExecutionRecord(
                execution_id=str(result.execution_id), trade_id=str(result.trade_id),
                symbol=result.symbol, direction=result.side,
                entry_price=signal.entry_price,
                fill_price=fill.fill_price if fill else None,
                slippage=fill.slippage if fill else None, qty=result.qty,
                stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                broker_state=result.broker_state, position_side=result.position_side,
                execution_fingerprint=result.execution_fingerprint,
                pipeline_version=settings.pipeline_version,
                strategy_version=settings.strategy_version,
                ai_prompt_version=settings.prompt_version,
                model_version=settings.model_version,
                created_at=result.submitted_at,
                filled_at=fill.fill_time if fill else None,
            ))
            s.commit()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/execution/test_executor.py -v
```

Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add app/execution/executor.py tests/execution/test_executor.py
git commit -m "feat: execution engine, bracket orders, idempotency, fill polling"
```

---

### Task 11: Logger + Report Generator

**Files:** `app/persistence/logger.py`, `tests/persistence/test_logger.py`

- [ ] **Step 1: Write failing tests**

`tests/persistence/test_logger.py`:
```python
import pytest
from datetime import datetime, date
from uuid import uuid4
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.persistence.db import Base, CycleRecord
from app.persistence.logger import PipelineLogger
from app.pipeline.context import PipelineContext
from app.models.market import MarketData
from app.models.signals import TradeRejection

TEST_DB = "sqlite:///data/test_logger.db"

@pytest.fixture
def logger():
    engine = create_engine(TEST_DB)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    @contextmanager
    def factory():
        s = Session(); yield s; s.close()
    yield PipelineLogger(session_factory=factory)
    Base.metadata.drop_all(engine)

def make_ctx():
    ctx = PipelineContext()
    ctx.market_data = MarketData(
        symbol="SPY", timestamp=datetime.utcnow(), pipeline_version="1.0.0",
        open=520.0, high=522.0, low=519.0, close=521.0, volume=1e6,
        bid=521.0, ask=521.1, spread_proxy=0.0001, vix=18.5, news_sentiment=0.1)
    ctx.signal = TradeRejection(symbol="SPY", strategy="spy_trend_following",
                                reasons=["test"], strategy_confidence=0.3)
    return ctx

def test_write_cycle_persists(logger):
    ctx = make_ctx()
    logger.write_cycle(ctx)
    from sqlalchemy import create_engine
    e2 = create_engine(TEST_DB)
    Session2 = sessionmaker(bind=e2)
    with Session2() as s:
        rec = s.get(CycleRecord, str(ctx.cycle_id))
        assert rec is not None

def test_report_file_created(logger, tmp_path, monkeypatch):
    monkeypatch.setattr("app.persistence.logger.REPORTS_DIR", str(tmp_path))
    ctx = make_ctx()
    logger.write_cycle(ctx)
    logger.write_report(ctx, report_date=date(2026, 5, 16))
    assert (tmp_path / "2026-05-16.md").exists()
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/persistence/test_logger.py -v
```

- [ ] **Step 3: Create app/persistence/logger.py**

```python
import os
from datetime import datetime, date
from app.pipeline.context import PipelineContext
from app.persistence.db import CycleRecord
from app.models.signals import TradeSignal, TradeRejection
from app.config import settings

REPORTS_DIR = "reports"

class PipelineLogger:
    def __init__(self, session_factory=None):
        self._sf = session_factory

    def write_cycle(self, ctx: PipelineContext) -> None:
        if not self._sf: return
        rec = CycleRecord(
            cycle_id=str(ctx.cycle_id),
            started_at=ctx.started_at,
            completed_at=datetime.utcnow(),
            trading_mode=ctx.trading_mode,
            pipeline_version=settings.pipeline_version,
            strategy_version=settings.strategy_version,
            ai_prompt_version=settings.prompt_version,
            model_version=settings.model_version,
            market_data_json=ctx.market_data.model_dump(mode="json") if ctx.market_data else None,
            features_json=ctx.features.model_dump(mode="json") if ctx.features else None,
            signal_json=ctx.signal.model_dump(mode="json") if ctx.signal else None,
            ai_analysis_json=ctx.ai_analysis.model_dump(mode="json") if ctx.ai_analysis else None,
            risk_json=ctx.risk.model_dump(mode="json") if ctx.risk else None,
            validation_json=ctx.validation.model_dump(mode="json") if ctx.validation else None,
            errors_json=ctx.errors,
        )
        with self._sf() as s:
            s.merge(rec); s.commit()

    def write_report(self, ctx: PipelineContext, report_date: date | None = None) -> None:
        d = report_date or date.today()
        os.makedirs(REPORTS_DIR, exist_ok=True)
        path = os.path.join(REPORTS_DIR, f"{d.isoformat()}.md")

        sig_line = "N/A"
        if isinstance(ctx.signal, TradeSignal):
            sig_line = (f"LONG @ ${ctx.signal.entry_price} | "
                        f"SL: ${ctx.signal.stop_loss} | TP: ${ctx.signal.take_profit} | "
                        f"conf: {ctx.signal.strategy_confidence:.2f}")
        elif isinstance(ctx.signal, TradeRejection):
            sig_line = f"NO TRADE — {', '.join(ctx.signal.reasons)}"

        ai_line = "N/A"
        if ctx.ai_analysis:
            ai_line = (f"{ctx.ai_analysis.decision} | "
                       f"ai_confidence: {ctx.ai_analysis.ai_confidence:.2f} | "
                       f"regime: {ctx.ai_analysis.regime}")

        risk_line = "N/A"
        if ctx.risk:
            risk_line = f"{ctx.risk.outcome} | risk_score: {ctx.risk.risk_score:.3f}"

        with open(path, "w") as f:
            f.write(f"""# Trading Report — {d.isoformat()}

## Market Assessment
Symbol: {ctx.market_data.symbol if ctx.market_data else 'N/A'} | VIX: {ctx.market_data.vix if ctx.market_data else '?'}

## Strategy Signal
{sig_line}

## AI Analysis
{ai_line}

## Risk Engine
{risk_line}

## Errors
{chr(10).join(ctx.errors) or 'None'}
""")
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/persistence/test_logger.py -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add app/persistence/logger.py tests/persistence/test_logger.py
git commit -m "feat: pipeline logger, atomic SQLite write, daily .md report"
```

---

### Task 12: Pipeline Orchestrator

**Files:** `app/pipeline/trading_pipeline.py`, `tests/pipeline/test_trading_pipeline.py`

- [ ] **Step 1: Write failing tests**

`tests/pipeline/test_trading_pipeline.py`:
```python
import pytest
from unittest.mock import MagicMock
from datetime import datetime
from uuid import uuid4
from app.pipeline.trading_pipeline import TradingPipeline
from app.pipeline.context import PipelineContext
from app.models.market import MarketData, FeatureSet
from app.models.signals import TradeSignal, TradeRejection
from app.models.ai import AIAnalysis
from app.models.risk import RiskDecision, ValidationResult

def stub_pipeline():
    p = TradingPipeline.__new__(TradingPipeline)
    p.fetcher   = MagicMock()
    p.strategy  = MagicMock()
    p.analyst   = MagicMock()
    p.risk      = MagicMock()
    p.validator = MagicMock()
    p.executor  = MagicMock()
    p.logger    = MagicMock()
    return p

def wire_happy(p):
    p.fetcher.fetch.return_value = MagicMock(spec=MarketData, vix=18.0, symbol="SPY",
        timestamp=datetime.utcnow(), pipeline_version="1.0.0",
        open=520.0, high=522.0, low=519.0, close=521.0, volume=1e6,
        bid=521.0, ask=521.1, spread_proxy=0.0001, news_sentiment=0.1, bars_daily=[])
    p.strategy.evaluate.return_value = MagicMock(spec=TradeSignal, type="SIGNAL",
        trade_id=uuid4(), strategy_confidence=0.74)
    p.analyst.analyze.return_value = MagicMock(spec=AIAnalysis, decision="APPROVE",
        ai_confidence=0.81, failed=False)
    p.risk.evaluate.return_value = MagicMock(spec=RiskDecision, outcome="APPROVED")
    p.validator.validate.return_value = MagicMock(spec=ValidationResult, outcome="PASS")
    p.executor.execute.return_value = None

def test_happy_path(mocker):
    p = stub_pipeline()
    wire_happy(p)
    # patch FeaturePipeline.compute
    mocker.patch("app.pipeline.trading_pipeline.FeaturePipeline.compute",
                 return_value=MagicMock(spec=FeatureSet, vix=18.0))
    ctx = p.run_assessment()
    assert isinstance(ctx, PipelineContext)
    p.logger.write_cycle.assert_called_once()

def test_fetch_error_logged(mocker):
    p = stub_pipeline()
    p.fetcher.fetch.side_effect = RuntimeError("API down")
    ctx = p.run_assessment()
    assert any("API down" in e for e in ctx.errors)
    p.logger.write_cycle.assert_called_once()

def test_kill_skips_execution(mocker):
    p = stub_pipeline()
    wire_happy(p)
    mocker.patch("app.pipeline.trading_pipeline.FeaturePipeline.compute",
                 return_value=MagicMock(spec=FeatureSet, vix=18.0))
    p.risk.evaluate.return_value = MagicMock(spec=RiskDecision, outcome="KILL")
    p.run_assessment()
    p.executor.execute.assert_not_called()
```

- [ ] **Step 2: Run — expect ImportError**

```bash
pytest tests/pipeline/test_trading_pipeline.py -v
```

- [ ] **Step 3: Create app/pipeline/trading_pipeline.py**

```python
import hashlib
from app.pipeline.context import PipelineContext
from app.data.fetcher import make_fetcher
from app.features.pipeline import FeaturePipeline
from app.strategies.spy_trend import SpyTrendStrategy
from app.ai.analyst import AIAnalyst
from app.risk.engine import RiskEngine
from app.validation.validator import TradeValidator
from app.execution.executor import ExecutionEngine
from app.persistence.logger import PipelineLogger
from app.models.signals import TradeSignal, TradeRejection
from app.config import settings

class TradingPipeline:
    def __init__(self, session_factory=None):
        self.fetcher   = make_fetcher()
        self.strategy  = SpyTrendStrategy()
        self.analyst   = AIAnalyst()
        self.risk      = RiskEngine(session_factory)
        self.validator = TradeValidator(session_factory)
        self.executor  = ExecutionEngine(session_factory)
        self.logger    = PipelineLogger(session_factory)

    def run_assessment(self) -> PipelineContext:
        ctx = PipelineContext(trading_mode=settings.trading_mode)

        try:
            ctx.market_data = self.fetcher.fetch(settings.trading_symbol)
        except Exception as e:
            ctx.errors.append(f"DataFetcher: {e}")
            self.logger.write_cycle(ctx); self.logger.write_report(ctx)
            return ctx

        try:
            ctx.features = FeaturePipeline.compute(ctx.market_data)
        except Exception as e:
            ctx.errors.append(f"FeaturePipeline: {e}")
            self.logger.write_cycle(ctx); self.logger.write_report(ctx)
            return ctx

        ctx.signal = self.strategy.evaluate(ctx.features)

        ctx.ai_analysis = self.analyst.analyze(ctx.signal, ctx.features)
        if ctx.ai_analysis.failed:
            ctx.errors.append("AI analysis failed")
            self.logger.write_cycle(ctx); self.logger.write_report(ctx)
            return ctx

        eval_signal = ctx.signal if isinstance(ctx.signal, TradeSignal) else _dummy_signal()
        ctx.risk = self.risk.evaluate(eval_signal, ctx.ai_analysis, vix=ctx.features.vix)

        if ctx.risk.outcome in ("KILL", "BLOCKED"):
            self.logger.write_cycle(ctx); self.logger.write_report(ctx)
            return ctx

        ctx.validation = self.validator.validate(ctx.signal)

        if ctx.validation.outcome == "PASS" and isinstance(ctx.signal, TradeSignal):
            fp = hashlib.sha256(
                (ctx.signal.model_dump_json() +
                 ctx.features.model_dump_json() +
                 ctx.ai_analysis.model_dump_json()).encode()
            ).hexdigest()
            ctx.order = self.executor.execute(ctx.signal, account_balance=100.0, fingerprint=fp)

        self.logger.write_cycle(ctx)
        self.logger.write_report(ctx)
        return ctx

def _dummy_signal() -> TradeSignal:
    from uuid import uuid4
    return TradeSignal(trade_id=uuid4(), symbol="SPY", strategy="spy_trend_following",
                       direction="long", strategy_confidence=0.0,
                       entry_price=0.0, stop_loss=0.0, take_profit=0.0)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/pipeline/test_trading_pipeline.py -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/trading_pipeline.py tests/pipeline/test_trading_pipeline.py
git commit -m "feat: pipeline orchestrator, linear flow, error isolation per stage"
```

---

### Task 13: APScheduler + main.py startup

**Files:** `app/scheduler.py`, `main.py` (update)

- [ ] **Step 1: Create app/scheduler.py**

```python
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_ERROR
import pandas_market_calendars as mcal
from datetime import date
import logging

logger = logging.getLogger(__name__)
_DEFAULTS = dict(max_instances=1, coalesce=True, misfire_grace_time=30)

def _market_open() -> bool:
    nyse = mcal.get_calendar("NYSE")
    return not nyse.schedule(start_date=date.today().isoformat(),
                              end_date=date.today().isoformat()).empty

def _guard(fn):
    def wrapper(*a, **kw):
        if not _market_open():
            logger.info("Market closed — skipping %s", fn.__name__)
            return
        return fn(*a, **kw)
    wrapper.__name__ = fn.__name__
    return wrapper

def _assessment_job():
    from app.pipeline.trading_pipeline import TradingPipeline
    ctx = TradingPipeline().run_assessment()
    logger.info("Cycle done: %s errors=%s", ctx.cycle_id, ctx.errors)

def _report_job():
    logger.info("Daily report job complete")

def create_scheduler() -> BackgroundScheduler:
    s = BackgroundScheduler(timezone="America/New_York")
    s.add_listener(lambda e: logger.error("Job failed: %s — %s", e.job_id, e.exception),
                   EVENT_JOB_ERROR)
    s.add_job(_guard(_assessment_job), "cron", hour=9, minute=45,
              id="assessment", **_DEFAULTS)
    s.add_job(_guard(_report_job), "cron", hour=16, minute=10,
              id="daily_report", **_DEFAULTS)
    return s
```

- [ ] **Step 2: Update main.py with scheduler wiring**

```python
from fastapi import FastAPI
from app.scheduler import create_scheduler
from app.config import settings

app = FastAPI(title="AI Trading Bot", version="0.1.0")
_scheduler = create_scheduler()

@app.on_event("startup")
def startup():
    _scheduler.start()

@app.on_event("shutdown")
def shutdown():
    _scheduler.shutdown()

@app.get("/health")
def health():
    return {"status": "ok", "mode": settings.trading_mode}
```

- [ ] **Step 3: Start server and verify no errors**

```bash
python scripts/seed_db.py
uvicorn main:app --reload
```

Expected: server starts, no exceptions in console.

- [ ] **Step 4: Commit**

```bash
git add app/scheduler.py main.py
git commit -m "feat: APScheduler with NYSE calendar guard, max_instances=1"
```

---

### Task 14: FastAPI Approval Endpoint

**Files:** `main.py` (update), `tests/test_api.py`

- [ ] **Step 1: Write failing tests**

`tests/test_api.py`:
```python
import pytest
from unittest.mock import patch, MagicMock
from uuid import uuid4
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    with patch("app.scheduler.create_scheduler") as m:
        m.return_value = MagicMock()
        import importlib, main as m2
        importlib.reload(m2)
        yield TestClient(m2.app)

def test_health(client):
    assert client.get("/health").status_code == 200

def test_approve_missing_trade(client):
    with patch("main._get_pending_trade", return_value=None):
        r = client.post(f"/approve/{uuid4()}")
        assert r.status_code == 404

def test_approve_executes(client):
    tid = uuid4()
    sig = {"type":"SIGNAL","trade_id":str(tid),"symbol":"SPY",
           "strategy":"spy_trend_following","direction":"long",
           "strategy_confidence":0.74,"entry_price":521.0,
           "stop_loss":516.8,"take_profit":527.3,"position_side":"long"}
    with patch("main._get_pending_trade", return_value=sig), \
         patch("main._execute_approved_trade", return_value={"status":"executed"}):
        r = client.post(f"/approve/{tid}")
        assert r.status_code == 200
```

- [ ] **Step 2: Run — expect failures**

```bash
pytest tests/test_api.py -v
```

- [ ] **Step 3: Add approval endpoint to main.py**

```python
from fastapi import FastAPI, HTTPException
from uuid import UUID
from app.scheduler import create_scheduler
from app.config import settings
import logging

logger = logging.getLogger(__name__)
app = FastAPI(title="AI Trading Bot", version="0.1.0")
_scheduler = create_scheduler()

@app.on_event("startup")
def startup(): _scheduler.start()

@app.on_event("shutdown")
def shutdown(): _scheduler.shutdown()

@app.get("/health")
def health(): return {"status": "ok", "mode": settings.trading_mode}

@app.post("/approve/{trade_id}")
def approve_trade(trade_id: UUID):
    data = _get_pending_trade(trade_id)
    if data is None:
        raise HTTPException(404, "Pending trade not found")
    return _execute_approved_trade(trade_id, data)

def _get_pending_trade(trade_id: UUID) -> dict | None:
    from app.persistence.db import PendingTradeRecord, make_engine, get_session
    with get_session(make_engine()) as s:
        rec = s.query(PendingTradeRecord).filter_by(
            id=str(trade_id), status="PENDING_APPROVAL").first()
        return rec.signal_json if rec else None

def _execute_approved_trade(trade_id: UUID, signal_data: dict) -> dict:
    from app.models.signals import TradeSignal
    from app.models.ai import AIAnalysis
    from app.risk.engine import RiskEngine
    from app.validation.validator import TradeValidator
    from app.execution.executor import ExecutionEngine
    from app.persistence.db import PendingTradeRecord, make_engine, get_session
    from datetime import datetime

    signal = TradeSignal(**signal_data)
    dummy = AIAnalysis(decision="APPROVE", ai_confidence=0.80, regime="bullish",
                       reasoning="Manual approval", risk_factors=[], no_trade_reasons=[],
                       raw_prompt="", raw_response="",
                       model_version=settings.model_version, prompt_version=settings.prompt_version)

    risk = RiskEngine().evaluate(signal, dummy, vix=20.0)
    if risk.outcome in ("KILL", "BLOCKED"):
        raise HTTPException(400, f"Risk engine blocked: {risk.reason}")

    val = TradeValidator()
    val._trading_mode = "paper_auto"
    if val.validate(signal).outcome != "PASS":
        raise HTTPException(400, "Validation failed")

    order = ExecutionEngine().execute(signal, account_balance=100.0)
    engine = make_engine()
    with get_session(engine) as s:
        rec = s.query(PendingTradeRecord).filter_by(id=str(trade_id)).first()
        if rec:
            rec.status = "EXECUTED" if order else "FAILED"
            rec.updated_at = datetime.utcnow()
            s.commit()
    return {"status": "executed" if order else "failed", "trade_id": str(trade_id)}
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_api.py -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_api.py
git commit -m "feat: /approve endpoint, re-validates before execution"
```

---

### Task 15: Streamlit Dashboard

**Files:** `app/dashboard/app.py`

No unit tests — verify by running and checking in browser.

- [ ] **Step 1: Create app/dashboard/app.py**

```python
import streamlit as st
import pandas as pd
from datetime import datetime, date
from sqlalchemy import create_engine, text
from app.config import settings
import httpx

st.set_page_config(page_title="AI Trading Bot", layout="wide")

@st.cache_resource
def _engine():
    return create_engine(settings.database_url)

def _q(sql, **params):
    with _engine().connect() as c:
        rows = c.execute(text(sql), params).fetchall()
    return [dict(r._mapping) for r in rows]

def _q1(sql, **params):
    with _engine().connect() as c:
        row = c.execute(text(sql), params).fetchone()
    return dict(row._mapping) if row else {}

# Header
summary = _q1("SELECT * FROM daily_summary WHERE date=:d", d=date.today().isoformat())
ks      = _q1("SELECT * FROM kill_switch_state WHERE id=1")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Mode",   settings.trading_mode.upper())
c2.metric("P&L",    f"${summary.get('total_pnl', 0.0):.2f}")
c3.metric("Kill",   "🔴 ON" if ks.get("active") else "🟢 OFF")
c4.markdown(f"**Updated:** {datetime.utcnow().strftime('%H:%M:%S UTC')}")
st.divider()

# Pending approval (live_manual only)
if settings.trading_mode == "live_manual":
    pending = _q("SELECT * FROM pending_trades WHERE status='PENDING_APPROVAL'")
    if pending:
        st.subheader("⏳ Pending Approval")
        for pt in pending:
            sig = pt.get("signal_json", {})
            with st.container(border=True):
                a, b, c = st.columns([3,1,1])
                a.markdown(
                    f"**{sig.get('symbol')} {sig.get('direction','').upper()}** "
                    f"@ ${sig.get('entry_price',0):.2f}  \n"
                    f"SL: ${sig.get('stop_loss',0):.2f} | TP: ${sig.get('take_profit',0):.2f}  \n"
                    f"conf: {sig.get('strategy_confidence',0):.2f}"
                )
                if b.button("✅ Approve", key=f"ap_{pt['id']}"):
                    try:
                        r = httpx.post(f"http://localhost:8000/approve/{pt['id']}", timeout=10)
                        if r.status_code == 200: st.rerun()
                        else: st.error(r.json().get("detail"))
                    except Exception as e:
                        st.error(str(e))
                c.button("❌ Reject", key=f"re_{pt['id']}")

# Latest cycle
cycle = _q1("SELECT * FROM cycles ORDER BY started_at DESC LIMIT 1")
if cycle:
    st.subheader("📊 Latest Cycle")
    sig  = cycle.get("signal_json") or {}
    ai   = cycle.get("ai_analysis_json") or {}
    risk = cycle.get("risk_json") or {}
    st.markdown(f"**Signal:** {sig.get('type','N/A')} | conf: {sig.get('strategy_confidence',0):.2f}")
    st.markdown(f"**AI:** {ai.get('decision','N/A')} | ai_conf: {ai.get('ai_confidence',0):.2f} | {ai.get('regime','N/A')}")
    with st.expander("AI Reasoning"):
        st.write(ai.get("reasoning", "None"))
    st.markdown(f"**Risk:** {risk.get('outcome','N/A')} | score: {risk.get('risk_score',0):.3f}")

# Trade history
st.subheader("📋 Trade History")
trades = _q("SELECT * FROM trade_executions ORDER BY created_at DESC LIMIT :n",
            n=settings.trade_history_page_size)
if trades:
    df = pd.DataFrame(trades)[["symbol","direction","entry_price","fill_price",
                                "slippage","broker_state","created_at"]]
    st.dataframe(df, use_container_width=True)
else:
    st.info("No trades yet.")

# Performance
st.subheader("📈 Performance (today)")
if summary:
    p1,p2,p3,p4 = st.columns(4)
    p1.metric("Win Rate",    f"{summary.get('win_rate',0)*100:.0f}%")
    p2.metric("Profit Factor", f"{summary.get('profit_factor',0):.2f}")
    p3.metric("Drawdown",    f"{summary.get('drawdown_pct',0)*100:.1f}%")
    p4.metric("Consec Losses", summary.get("consecutive_losses", 0))

import time
time.sleep(settings.dashboard_refresh_seconds)
st.rerun()
```

- [ ] **Step 2: Run dashboard**

```bash
streamlit run app/dashboard/app.py
```

Expected: browser opens, all sections visible, no Python errors.

- [ ] **Step 3: Commit**

```bash
git add app/dashboard/app.py
git commit -m "feat: Streamlit dashboard, lazy AI reasoning, paginated history"
```

---

### Task 16: Replay Script

**Files:** `scripts/replay_day.py`

- [ ] **Step 1: Create scripts/replay_day.py**

```python
"""
Replay a stored pipeline cycle for debugging and prompt tuning.

Usage:
  python scripts/replay_day.py --date 2026-05-16
  python scripts/replay_day.py --cycle-id <uuid>
  python scripts/replay_day.py --date 2026-05-16 --mode ai_only
"""
import argparse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.persistence.db import CycleRecord
from app.config import settings

def load_cycle(engine, date_str, cycle_id):
    Session = sessionmaker(bind=engine)
    with Session() as s:
        if cycle_id:
            return s.get(CycleRecord, cycle_id)
        if date_str:
            return s.query(CycleRecord).filter(
                CycleRecord.started_at >= f"{date_str}T00:00:00",
                CycleRecord.started_at <  f"{date_str}T23:59:59",
            ).order_by(CycleRecord.started_at.desc()).first()
    return None

def replay(cycle: CycleRecord, mode: str) -> None:
    print(f"\n=== Replay {cycle.cycle_id} | mode={mode} ===\n")

    if mode == "feature_only":
        from app.models.market import MarketData
        from app.features.pipeline import FeaturePipeline
        fs = FeaturePipeline.compute(MarketData(**cycle.market_data_json))
        print(fs.model_dump_json(indent=2))

    elif mode == "strategy_only":
        from app.models.market import FeatureSet
        from app.strategies.spy_trend import SpyTrendStrategy
        result = SpyTrendStrategy.evaluate(FeatureSet(**cycle.features_json))
        print(result.model_dump_json(indent=2))

    elif mode == "ai_only":
        from app.models.market import FeatureSet
        from app.models.signals import TradeSignal, TradeRejection
        from app.ai.analyst import AIAnalyst
        fs  = FeatureSet(**cycle.features_json)
        sig_data = cycle.signal_json
        sig = (TradeSignal(**sig_data) if sig_data.get("type") == "SIGNAL"
               else TradeRejection(**sig_data))
        print(AIAnalyst().analyze(sig, fs).model_dump_json(indent=2))

    elif mode == "full_pipeline":
        from app.models.market import MarketData
        from app.features.pipeline import FeaturePipeline
        from app.strategies.spy_trend import SpyTrendStrategy
        from app.ai.analyst import AIAnalyst
        md  = MarketData(**cycle.market_data_json)
        fs  = FeaturePipeline.compute(md)
        sig = SpyTrendStrategy.evaluate(fs)
        print(AIAnalyst().analyze(sig, fs).model_dump_json(indent=2))

    print("\n=== Done ===")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date")
    p.add_argument("--cycle-id")
    p.add_argument("--mode", default="full_pipeline",
                   choices=["full_pipeline","feature_only","strategy_only","ai_only"])
    args = p.parse_args()

    engine = create_engine(settings.database_url)
    cycle = load_cycle(engine, args.date, getattr(args, "cycle_id", None))
    if not cycle:
        print("No cycle found."); return
    replay(cycle, args.mode)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script runs**

```bash
python scripts/replay_day.py --date 2026-05-16 --mode feature_only
```

Expected: `No cycle found.` (no data yet — script runs without crashing).

- [ ] **Step 3: Commit**

```bash
git add scripts/replay_day.py
git commit -m "feat: replay script with feature_only/strategy_only/ai_only/full_pipeline"
```

---

## Final Verification

- [ ] **Run full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests PASS, zero failures.

- [ ] **Seed and start server**

```bash
python scripts/seed_db.py
uvicorn main:app --reload
curl http://localhost:8000/health
```

Expected: `{"status":"ok","mode":"dry_run"}`

- [ ] **Smoke test dry_run cycle**

```python
from app.pipeline.trading_pipeline import TradingPipeline
ctx = TradingPipeline().run_assessment()
print("Signal:", ctx.signal)
print("Errors:", ctx.errors)
```

Expected: `TradeSignal` or `TradeRejection` printed, no unhandled exceptions.

- [ ] **Verify .md report written**

```bash
ls reports/
```

Expected: `2026-05-16.md` (or today's date).

- [ ] **Run dashboard**

```bash
streamlit run app/dashboard/app.py
```

Expected: dashboard opens in browser, all sections render.

- [ ] **Final commit**

```bash
git add .
git commit -m "feat: P1 Core Engine complete — paper trading pipeline with Claude AI"
```
