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
