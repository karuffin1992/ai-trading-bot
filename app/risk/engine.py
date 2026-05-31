import logging
from app.models.signals import TradeSignal
from app.models.ai import AIAnalysis
from app.models.risk import RiskDecision, KillSwitchState
from app.config import settings
from app.util.clock import now_utc, now_et, et_day_start_utc, et_today_iso

logger = logging.getLogger(__name__)

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
        if not self._stop_loss_valid(signal):
            return RiskDecision(outcome="BLOCKED", reason="Stop loss missing or invalid",
                                tier="hard_block")
        balance = self._get_account_balance()
        if balance < settings.min_balance_threshold:
            return RiskDecision(outcome="BLOCKED", reason=f"Balance ${balance:.2f} below floor",
                                tier="hard_block")
        if self._get_daily_trade_count() >= settings.max_daily_trades:
            return RiskDecision(outcome="BLOCKED", reason="Daily trade limit", tier="hard_block")
        if self._has_open_position(signal.symbol):
            return RiskDecision(outcome="BLOCKED", reason=f"Open position in {signal.symbol}",
                                tier="hard_block")
        if self._is_earnings_proximity(signal.symbol):
            return RiskDecision(outcome="BLOCKED", reason="Earnings proximity", tier="hard_block")
        last = self._get_last_trade_time()
        if last:
            elapsed = (now_utc() - last).total_seconds() / 60
            if elapsed < settings.cooldown_minutes:
                return RiskDecision(outcome="BLOCKED",
                                    reason=f"Cooldown {settings.cooldown_minutes-elapsed:.0f}m",
                                    tier="hard_block")

        # Tier 3: Composite Score
        vol_score = max(0.0, 1.0 - vix / settings.vix_max)
        score = round(signal.strategy_confidence*0.40 +
                      analysis.ai_confidence*0.35 +
                      vol_score*0.25, 4)
        if score < settings.risk_score_threshold:
            return RiskDecision(outcome="BLOCKED",
                                reason=f"risk_score {score:.3f} < {settings.risk_score_threshold}",
                                risk_score=score, tier="score")

        return RiskDecision(outcome="APPROVED", reason="All checks passed",
                            risk_score=score, tier="none")

    @staticmethod
    def _stop_loss_valid(signal: TradeSignal) -> bool:
        if not signal.stop_loss or signal.stop_loss <= 0:
            return False
        if signal.direction == "long":
            return signal.stop_loss < signal.entry_price
        return signal.stop_loss > signal.entry_price

    def _get_account_balance(self) -> float:
        from alpaca.trading.client import TradingClient
        return float(TradingClient(settings.alpaca_api_key, settings.alpaca_secret_key,
                                   paper=True).get_account().cash)

    def _get_daily_trade_count(self) -> int:
        if not self._sf: return 0
        from app.persistence.db import TradeExecutionRecord
        with self._sf() as s:
            return s.query(TradeExecutionRecord).filter(
                TradeExecutionRecord.created_at >= et_day_start_utc(),
                TradeExecutionRecord.broker_state == "filled").count()

    def _get_daily_drawdown(self) -> float:
        if not self._sf: return 0.0
        from app.persistence.db import DailySummaryRecord
        with self._sf() as s:
            r = s.get(DailySummaryRecord, et_today_iso())
            return r.drawdown_pct if r else 0.0

    def _get_consecutive_losses(self) -> int:
        if not self._sf: return 0
        from app.persistence.db import DailySummaryRecord
        with self._sf() as s:
            r = s.get(DailySummaryRecord, et_today_iso())
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
                ks.active, ks.reason, ks.activated_at = True, reason, now_utc()
                s.commit()

    def _get_last_trade_time(self):
        if not self._sf: return None
        from app.persistence.db import TradeExecutionRecord
        with self._sf() as s:
            r = s.query(TradeExecutionRecord).order_by(
                TradeExecutionRecord.created_at.desc()).first()
            return r.created_at if r else None

    def _has_open_position(self, symbol: str) -> bool:
        try:
            from alpaca.trading.client import TradingClient
            positions = TradingClient(settings.alpaca_api_key, settings.alpaca_secret_key,
                                      paper=True).get_all_positions()
            return any(p.symbol == symbol for p in positions)
        except Exception:
            return False

    def _is_earnings_proximity(self, symbol: str) -> bool:
        try:
            import finnhub
            from datetime import timedelta
            client = finnhub.Client(api_key=settings.finnhub_api_key)
            today = now_et().date()
            horizon = today + timedelta(days=settings.earnings_proximity_days)
            cal = client.earnings_calendar(_from=today.isoformat(),
                                           to=horizon.isoformat(), symbol=symbol)
            return bool(cal.get("earningsCalendar"))
        except Exception:
            return False
