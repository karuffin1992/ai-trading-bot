from uuid import uuid4
from app.config import settings
from app.util.clock import now_utc
from app.persistence.summary import compute_daily_summary

# Owns the open->closed lifecycle of positions. Executor records the open leg;
# the scheduler's monitor job calls sync() to detect broker-side closes (bracket
# TP/SL fills) and EOD close_all() to flatten. Every close recomputes the daily
# summary so the RiskEngine sees fresh drawdown / loss state.
class PositionManager:
    def __init__(self, session_factory=None):
        self._sf = session_factory
        self._client = None

    def _broker(self):
        if self._client is None:
            from alpaca.trading.client import TradingClient
            self._client = TradingClient(settings.alpaca_api_key,
                                         settings.alpaca_secret_key, paper=True)
        return self._client

    def record_open(self, trade_id, symbol, direction, qty,
                    entry_price, stop_loss, take_profit) -> None:
        if not self._sf:
            return
        from app.persistence.db import PositionRecord
        with self._sf() as s:
            s.add(PositionRecord(
                id=str(uuid4()), trade_id=str(trade_id), symbol=symbol,
                direction=direction, qty=qty, entry_price=entry_price,
                stop_loss=stop_loss, take_profit=take_profit,
                status="open", pnl=0.0, opened_at=now_utc(), closed_at=None,
            ))
            s.commit()

    def sync(self) -> None:
        if not self._sf:
            return
        from app.persistence.db import PositionRecord
        open_at_broker = self._broker_open_map()
        with self._sf() as s:
            rows = s.query(PositionRecord).filter_by(status="open").all()
            changed = False
            for r in rows:
                bp = open_at_broker.get(r.symbol)
                if bp is not None:
                    r.pnl = bp["unrealized_pl"]
                else:
                    r.pnl = self._realized_pnl(r)
                    r.status = "closed"
                    r.closed_at = now_utc()
                    changed = True
            s.commit()
            if changed:
                compute_daily_summary(s)

    def close_all(self) -> None:
        if not self._sf:
            return
        from app.persistence.db import PositionRecord
        with self._sf() as s:
            rows = s.query(PositionRecord).filter_by(status="open").all()
            for r in rows:
                exit_price = self._flatten(r.symbol)
                r.pnl = self._pnl(r.direction, r.entry_price, exit_price, r.qty)
                r.status = "closed"
                r.closed_at = now_utc()
            s.commit()
            compute_daily_summary(s)

    def _broker_open_map(self) -> dict:
        try:
            positions = self._broker().get_all_positions()
            return {p.symbol: {"unrealized_pl": float(p.unrealized_pl),
                               "current_price": float(p.current_price)}
                    for p in positions}
        except Exception:
            return {}

    def _realized_pnl(self, record) -> float:
        exit_price = self._last_price(record.symbol) or record.entry_price
        return self._pnl(record.direction, record.entry_price, exit_price, record.qty)

    def _flatten(self, symbol: str) -> float:
        try:
            self._broker().close_position(symbol)
        except Exception:
            pass
        return self._last_price(symbol) or 0.0

    def _last_price(self, symbol: str) -> float | None:
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest
            c = StockHistoricalDataClient(settings.alpaca_api_key, settings.alpaca_secret_key)
            q = c.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))[symbol]
            return (float(q.bid_price) + float(q.ask_price)) / 2
        except Exception:
            return None

    @staticmethod
    def _pnl(direction: str, entry: float, exit_price: float, qty: float) -> float:
        sign = 1.0 if direction in ("long", "buy") else -1.0
        return round(sign * (exit_price - entry) * qty, 4)
