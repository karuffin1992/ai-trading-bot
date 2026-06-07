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
    # Self-learning / memory layer. All default OFF: with these unset the
    # pipeline (and AIAnalyst prompt) behaves exactly as before.
    memory_enabled: bool = False
    memory_injection_enabled: bool = False
    memory_retrieval_k: int = 5
    memory_token_budget: int = 800
    embedding_provider: str = "deterministic_hash"
    embedding_dim: int = 256
    embedding_version: str = "det-1.0.0"
    llm_provider: str = "claude"
    llm_replay_cache_enabled: bool = True
    reflection_enabled: bool = False

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
    # Optional blocks — read with .get() so pre-existing YAML files (without these
    # sections) still load without KeyError.
    mem = d.get("memory", {})
    llm = d.get("llm", {})
    refl = d.get("reflection", {})
    optional = {
        "MODEL_VERSION": d.get("versions", {}).get("model"),
        "MEMORY_ENABLED": mem.get("enabled"),
        "MEMORY_INJECTION_ENABLED": mem.get("injection_enabled"),
        "MEMORY_RETRIEVAL_K": mem.get("retrieval_k"),
        "MEMORY_TOKEN_BUDGET": mem.get("token_budget"),
        "EMBEDDING_PROVIDER": mem.get("embedding_provider"),
        "EMBEDDING_DIM": mem.get("embedding_dim"),
        "EMBEDDING_VERSION": mem.get("embedding_version"),
        "LLM_PROVIDER": llm.get("provider"),
        "LLM_REPLAY_CACHE_ENABLED": llm.get("replay_cache_enabled"),
        "REFLECTION_ENABLED": refl.get("enabled"),
    }
    for k, v in optional.items():
        if v is not None:
            flat[k] = v
    for k, v in flat.items():
        os.environ.setdefault(k, str(v))

_apply_yaml()
settings = Settings()
