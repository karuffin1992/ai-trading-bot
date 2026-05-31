from fastapi import FastAPI, HTTPException
from uuid import UUID
from app.scheduler import create_scheduler
from app.config import settings
import logging

logger = logging.getLogger(__name__)
app = FastAPI(title="AI Trading Bot", version="0.1.0")
_scheduler = create_scheduler()

@app.on_event("startup")
def startup(): _scheduler.start()

@app.on_event("shutdown")
def shutdown(): _scheduler.shutdown()

@app.get("/health")
def health(): return {"status": "ok", "mode": settings.trading_mode}

@app.post("/approve/{trade_id}")
def approve_trade(trade_id: UUID):
    data = _get_pending_trade(trade_id)
    if data is None:
        raise HTTPException(404, "Pending trade not found")
    return _execute_approved_trade(trade_id, data)

def _get_pending_trade(trade_id: UUID) -> dict | None:
    from app.persistence.db import PendingTradeRecord, make_engine, get_session
    with get_session(make_engine()) as s:
        rec = s.query(PendingTradeRecord).filter_by(
            id=str(trade_id), status="PENDING_APPROVAL").first()
        return rec.signal_json if rec else None

def _execute_approved_trade(trade_id: UUID, signal_data: dict) -> dict:
    from app.models.signals import TradeSignal
    from app.models.ai import AIAnalysis
    from app.risk.engine import RiskEngine
    from app.validation.validator import TradeValidator
    from app.execution.executor import ExecutionEngine
    from app.persistence.db import PendingTradeRecord, make_engine, get_session
    from datetime import datetime

    signal = TradeSignal(**signal_data)
    dummy = AIAnalysis(decision="APPROVE", ai_confidence=0.80, regime="bullish",
                       reasoning="Manual approval", risk_factors=[], no_trade_reasons=[],
                       raw_prompt="", raw_response="",
                       model_version=settings.model_version, prompt_version=settings.prompt_version)

    risk = RiskEngine().evaluate(signal, dummy, vix=20.0)
    if risk.outcome in ("KILL", "BLOCKED"):
        raise HTTPException(400, f"Risk engine blocked: {risk.reason}")

    val = TradeValidator()
    val._trading_mode = "paper_auto"
    if val.validate(signal).outcome != "PASS":
        raise HTTPException(400, "Validation failed")

    order = ExecutionEngine().execute(signal, account_balance=100.0)
    engine = make_engine()
    with get_session(engine) as s:
        rec = s.query(PendingTradeRecord).filter_by(id=str(trade_id)).first()
        if rec:
            rec.status = "EXECUTED" if order else "FAILED"
            rec.updated_at = datetime.utcnow()
            s.commit()
    return {"status": "executed" if order else "failed", "trade_id": str(trade_id)}
