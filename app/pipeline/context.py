from dataclasses import dataclass, field
from uuid import UUID, uuid4
from datetime import datetime
from typing import Optional
from app.models.market import MarketData, FeatureSet
from app.models.signals import TradeSignal, TradeRejection
from app.models.ai import AIAnalysis
from app.models.risk import RiskDecision, ValidationResult
from app.models.execution import OrderResult

@dataclass
class PipelineContext:
    cycle_id: UUID = field(default_factory=uuid4)
    started_at: datetime = field(default_factory=datetime.utcnow)
    trading_mode: str = "dry_run"
    market_data: Optional[MarketData] = None
    features: Optional[FeatureSet] = None
    signal: Optional[TradeSignal | TradeRejection] = None
    ai_analysis: Optional[AIAnalysis] = None
    risk: Optional[RiskDecision] = None
    validation: Optional[ValidationResult] = None
    order: Optional[OrderResult] = None
    errors: list[str] = field(default_factory=list)
