import math
import pandas as pd
import pandas_ta_classic as ta
from datetime import datetime
from app.models.market import MarketData, FeatureSet
from app.config import settings

# pandas-ta-classic returns None (not a NaN Series) when the input has fewer
# rows than the indicator window; mirror original pandas-ta by yielding NaN.
def _last(series) -> float:
    return float(series.iloc[-1]) if series is not None else float("nan")

class FeaturePipeline:
    @staticmethod
    def compute(market_data: MarketData) -> FeatureSet:
        df = FeaturePipeline._to_df(market_data.bars_daily)
        close = df["close"]

        ema_9  = _last(ta.ema(close, length=9))
        ema_20 = _last(ta.ema(close, length=20))
        ema_50 = _last(ta.ema(close, length=50))
        rsi    = _last(ta.rsi(close, length=14))

        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_df is None:
            macd = macd_signal = macd_hist = float("nan")
        else:
            macd        = _last(macd_df["MACD_12_26_9"])
            macd_signal = _last(macd_df["MACDs_12_26_9"])
            macd_hist   = _last(macd_df["MACDh_12_26_9"])

        atr = _last(ta.atr(df["high"], df["low"], close, length=14))

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
