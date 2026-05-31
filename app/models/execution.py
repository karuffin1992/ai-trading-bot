from pydantic import BaseModel
from typing import Literal
from datetime import datetime
from uuid import UUID

class OrderResult(BaseModel):
    execution_id: UUID
    trade_id: UUID
    broker_order_id: str
    symbol: str
    qty: float
    side: Literal["buy", "sell"]
    position_side: Literal["long", "short"]
    submitted_at: datetime
    broker_state: Literal["submitted","filled","partial_fill","canceled","reconciled","stale"] = "submitted"
    execution_fingerprint: str = ""

class FillRecord(BaseModel):
    execution_id: UUID
    fill_price: float
    fill_time: datetime
    slippage: float
    broker_state: Literal["filled","partial_fill","stale","reconciled"]
