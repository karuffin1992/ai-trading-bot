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

# ---------------------------------------------------------------------------
# Self-learning / memory layer tables. Created automatically by
# Base.metadata.create_all (no Alembic). All rows are versioned for
# reproducibility; the learning_runs / parameter_sets / experiment_history
# tables are written by the deferred harness+learning layer but defined here now
# so the schema contract is locked.
# ---------------------------------------------------------------------------

class MemoryEpisodeRecord(Base):
    __tablename__ = "memory_episodes"
    episode_id = Column(String, primary_key=True)
    kind = Column(String)
    symbol = Column(String)
    cycle_id = Column(String)
    trade_id = Column(String)
    summary = Column(String)
    payload_json = Column(JSON)
    outcome = Column(String)
    pnl_pct = Column(Float)
    holding_time_minutes = Column(Integer)
    regime = Column(String)
    tags_json = Column(JSON)
    reflection = Column(String)
    embedding_id = Column(String)
    feature_version = Column(String)
    strategy_version = Column(String)
    prompt_version = Column(String)
    model_version = Column(String)
    created_at = Column(DateTime)

class MemoryReflectionRecord(Base):
    __tablename__ = "memory_reflections"
    reflection_id = Column(String, primary_key=True)
    trade_id = Column(String)
    symbol = Column(String)
    direction = Column(String)
    entry_price = Column(Float)
    exit_price = Column(Float)
    pnl = Column(Float)
    outcome = Column(String)
    regime = Column(String)
    summary = Column(String)
    strengths_json = Column(JSON)
    weaknesses_json = Column(JSON)
    regime_notes_json = Column(JSON)
    execution_notes_json = Column(JSON)
    tags_json = Column(JSON)
    feature_snapshot_json = Column(JSON)
    feature_version = Column(String)
    strategy_version = Column(String)
    prompt_version = Column(String)
    model_version = Column(String)
    created_at = Column(DateTime)

class EmbeddingRegistryRecord(Base):
    __tablename__ = "embedding_registry"
    embedding_id = Column(String, primary_key=True)
    episode_id = Column(String)
    provider = Column(String, default="deterministic_hash")
    dim = Column(Integer)
    vector_json = Column(JSON)
    content_hash = Column(String)
    embedding_version = Column(String)
    created_at = Column(DateTime)

class PromptVersionRecord(Base):
    __tablename__ = "prompt_versions"
    prompt_version = Column(String, primary_key=True)
    name = Column(String)
    content = Column(String)
    content_hash = Column(String)
    notes = Column(String)
    created_at = Column(DateTime)

class ModelRegistryRecord(Base):
    __tablename__ = "model_registry"
    model_version = Column(String, primary_key=True)
    provider = Column(String)
    display_name = Column(String)
    params_json = Column(JSON)
    notes = Column(String)
    created_at = Column(DateTime)

class LearningRunRecord(Base):
    __tablename__ = "learning_runs"
    run_id = Column(String, primary_key=True)
    run_type = Column(String)
    status = Column(String, default="running")
    config_json = Column(JSON)
    metrics_json = Column(JSON)
    dataset_fingerprint = Column(String)
    feature_version = Column(String)
    strategy_version = Column(String)
    prompt_version = Column(String)
    model_version = Column(String)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

class ParameterSetRecord(Base):
    __tablename__ = "parameter_sets"
    param_set_id = Column(String, primary_key=True)
    run_id = Column(String)
    status = Column(String, default="candidate")
    params_json = Column(JSON)
    score = Column(Float)
    metrics_json = Column(JSON)
    base_param_set_id = Column(String)
    created_at = Column(DateTime)

class ExperimentHistoryRecord(Base):
    __tablename__ = "experiment_history"
    experiment_id = Column(String, primary_key=True)
    run_id = Column(String)
    variant = Column(String)
    param_set_id = Column(String)
    dataset_fingerprint = Column(String)
    metrics_json = Column(JSON)
    passed_safety = Column(Boolean, default=False)
    created_at = Column(DateTime)

def make_engine(url: str | None = None):
    from sqlalchemy import create_engine
    return create_engine(url or settings.database_url)

def init_db(engine=None) -> None:
    import os
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    engine = engine or make_engine()
    Base.metadata.create_all(engine)
    with get_session(engine) as s:
        if not s.get(KillSwitchRecord, 1):
            s.add(KillSwitchRecord(id=1, active=False, reason=""))
            s.commit()

def make_session_factory(engine=None):
    engine = engine or make_engine()
    return lambda: get_session(engine)

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
