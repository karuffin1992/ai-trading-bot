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
        if self._trading_mode == "live_auto":
            return ValidationResult(outcome="FAIL", reason="live_auto unsupported in P1")

        spread = self._get_spread(signal.symbol)
        if spread > settings.max_spread_pct:
            return ValidationResult(outcome="FAIL",
                                    reason=f"spread {spread:.4%} > max {settings.max_spread_pct:.4%}")

        power = self._get_buying_power()
        if power < settings.min_balance_threshold:
            return ValidationResult(outcome="FAIL",
                                    reason=f"buying power ${power:.2f} below floor "
                                           f"${settings.min_balance_threshold:.2f}")

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
