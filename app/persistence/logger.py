import os
from datetime import date
from app.pipeline.context import PipelineContext
from app.persistence.db import CycleRecord
from app.models.signals import TradeSignal, TradeRejection
from app.config import settings
from app.util.clock import now_utc

REPORTS_DIR = "reports"

class PipelineLogger:
    def __init__(self, session_factory=None):
        self._sf = session_factory

    def write_cycle(self, ctx: PipelineContext) -> None:
        if not self._sf: return
        rec = CycleRecord(
            cycle_id=str(ctx.cycle_id),
            started_at=ctx.started_at,
            completed_at=now_utc(),
            trading_mode=ctx.trading_mode,
            pipeline_version=settings.pipeline_version,
            strategy_version=settings.strategy_version,
            ai_prompt_version=settings.prompt_version,
            model_version=settings.model_version,
            market_data_json=ctx.market_data.model_dump(mode="json") if ctx.market_data else None,
            features_json=ctx.features.model_dump(mode="json") if ctx.features else None,
            signal_json=ctx.signal.model_dump(mode="json") if ctx.signal else None,
            ai_analysis_json=ctx.ai_analysis.model_dump(mode="json") if ctx.ai_analysis else None,
            risk_json=ctx.risk.model_dump(mode="json") if ctx.risk else None,
            validation_json=ctx.validation.model_dump(mode="json") if ctx.validation else None,
            errors_json=ctx.errors,
        )
        with self._sf() as s:
            s.merge(rec); s.commit()

    def write_report(self, ctx: PipelineContext, report_date: date | None = None) -> None:
        d = report_date or date.today()
        os.makedirs(REPORTS_DIR, exist_ok=True)
        path = os.path.join(REPORTS_DIR, f"{d.isoformat()}.md")

        sig_line = "N/A"
        if isinstance(ctx.signal, TradeSignal):
            sig_line = (f"LONG @ ${ctx.signal.entry_price} | "
                        f"SL: ${ctx.signal.stop_loss} | TP: ${ctx.signal.take_profit} | "
                        f"conf: {ctx.signal.strategy_confidence:.2f}")
        elif isinstance(ctx.signal, TradeRejection):
            sig_line = f"NO TRADE — {', '.join(ctx.signal.reasons)}"

        ai_line = "N/A"
        if ctx.ai_analysis:
            ai_line = (f"{ctx.ai_analysis.decision} | "
                       f"ai_confidence: {ctx.ai_analysis.ai_confidence:.2f} | "
                       f"regime: {ctx.ai_analysis.regime}")

        risk_line = "N/A"
        if ctx.risk:
            risk_line = f"{ctx.risk.outcome} | risk_score: {ctx.risk.risk_score:.3f}"

        with open(path, "w") as f:
            f.write(f"""# Trading Report — {d.isoformat()}

## Market Assessment
Symbol: {ctx.market_data.symbol if ctx.market_data else 'N/A'} | VIX: {ctx.market_data.vix if ctx.market_data else '?'}

## Strategy Signal
{sig_line}

## AI Analysis
{ai_line}

## Risk Engine
{risk_line}

## Errors
{chr(10).join(ctx.errors) or 'None'}
""")
