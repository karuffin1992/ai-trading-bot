from pydantic import BaseModel, Field
from typing import Literal
from uuid import UUID, uuid4

class TradeSignal(BaseModel):
    type: Literal["SIGNAL"] = "SIGNAL"
    trade_id: UUID = Field(default_factory=uuid4)
    symbol: str
    strategy: str
    direction: Literal["long", "short"]
    strategy_confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_side: Literal["long", "short"] = "long"

class TradeRejection(BaseModel):
    type: Literal["REJECTION"] = "REJECTION"
    symbol: str
    strategy: str
    reasons: list[str]
    strategy_confidence: float
