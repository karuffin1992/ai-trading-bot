from pydantic import BaseModel
from typing import Literal

class AIAnalysis(BaseModel):
    decision: Literal["APPROVE", "REJECT", "REDUCE_CONFIDENCE", "NO_TRADE"]
    ai_confidence: float
    regime: Literal["bullish", "bearish", "neutral", "uncertain"]
    reasoning: str
    risk_factors: list[str]
    no_trade_reasons: list[str]
    raw_prompt: str
    raw_response: str
    model_version: str
    prompt_version: str
    failed: bool = False
