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
