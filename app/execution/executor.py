import time, logging
from uuid import uuid4, UUID
from app.models.signals import TradeSignal
from app.models.execution import OrderResult, FillRecord
from app.execution.position_manager import PositionManager
from app.config import settings
from app.util.clock import now_utc

logger = logging.getLogger(__name__)

class ExecutionEngine:
    def __init__(self, session_factory=None):
        self._sf = session_factory
        self._trading_mode = settings.trading_mode
        self._positions = PositionManager(session_factory)
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
        except Exception as e:
            logger.warning("Bracket submit failed (%s); falling back to market+stop", e)
            order = self._submit_market_fallback(signal, qty)
            if order is None:
                return None

        result = OrderResult(
            execution_id=execution_id, trade_id=signal.trade_id,
            broker_order_id=str(order.id), symbol=signal.symbol,
            qty=qty, side="buy", position_side=signal.position_side,
            submitted_at=now_utc(), execution_fingerprint=fingerprint,
        )

        fill = self._poll_fill(str(order.id), signal.entry_price)
        if fill:
            result.broker_state = fill.broker_state
        else:
            result.broker_state = self._reconcile(str(order.id))

        self._persist_execution(result, fill, signal)

        if result.broker_state in ("filled", "partial_fill"):
            fill_price = fill.fill_price if fill else signal.entry_price
            self._positions.record_open(
                signal.trade_id, signal.symbol, signal.position_side, qty,
                fill_price, signal.stop_loss, signal.take_profit)

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

    # Bracket unavailable: place the market entry, then attach a stop-loss after
    # fill. If the stop-loss cannot be placed, flatten the position immediately —
    # an unprotected position is worse than a missed trade.
    def _submit_market_fallback(self, signal: TradeSignal, qty: float):
        from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, StopOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        try:
            order = self._trading_client.submit_order(MarketOrderRequest(
                symbol=signal.symbol, qty=qty, side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY))
        except Exception as e:
            logger.error("Market fallback entry failed: %s", e)
            return None

        self._poll_fill(str(order.id), signal.entry_price)
        try:
            self._trading_client.submit_order(StopOrderRequest(
                symbol=signal.symbol, qty=qty, side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                stop_price=round(signal.stop_loss, 2)))
        except Exception as e:
            logger.critical("Stop-loss placement failed for %s; flattening: %s", signal.symbol, e)
            try:
                self._trading_client.close_position(signal.symbol)
            except Exception as ce:
                logger.critical("Emergency flatten failed for %s: %s", signal.symbol, ce)
            return None
        return order

    def _poll_fill(self, order_id: str, entry_price: float) -> FillRecord | None:
        deadline = time.time() + settings.fill_poll_timeout_seconds
        while time.time() < deadline:
            try:
                o = self._trading_client.get_order_by_id(order_id)
                if o.status.value == "filled":
                    fp = float(o.filled_avg_price or entry_price)
                    return FillRecord(execution_id=uuid4(), fill_price=fp,
                                      fill_time=now_utc(),
                                      slippage=fp - entry_price, broker_state="filled")
            except Exception:
                pass
            time.sleep(2)
        return None

    # Poll timed out: ask the broker for the order's terminal state instead of
    # blindly marking it stale, so the DB reflects reality.
    def _reconcile(self, order_id: str) -> str:
        try:
            o = self._trading_client.get_order_by_id(order_id)
            status = o.status.value
            if status in ("filled", "partial_fill", "canceled"):
                return "reconciled"
        except Exception as e:
            logger.warning("Reconcile failed for %s: %s", order_id, e)
        return "stale"

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
