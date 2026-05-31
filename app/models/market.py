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
