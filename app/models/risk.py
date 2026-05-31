from pydantic import BaseModel
from typing import Literal
from datetime import datetime

class KillSwitchState(BaseModel):
    active: bool = False
    reason: str = ""
    activated_at: datetime | None = None

class RiskDecision(BaseModel):
    outcome: Literal["APPROVED", "BLOCKED", "KILL"]
    reason: str
    risk_score: float = 0.0
    tier: Literal["kill_switch", "hard_block", "score", "none"] = "none"

class ValidationResult(BaseModel):
    outcome: Literal["PASS", "FAIL"]
    reason: str
