import json, os
import anthropic
from app.models.signals import TradeSignal, TradeRejection
from app.models.market import FeatureSet
from app.models.ai import AIAnalysis
from app.config import settings

_PROMPTS = os.path.join(os.path.dirname(__file__), "..", "prompts")

def _load(name: str) -> str:
    with open(os.path.join(_PROMPTS, name)) as f:
        return f.read()

class AIAnalyst:
    def __init__(self):
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def analyze(self, signal: TradeSignal | TradeRejection,
                features: FeatureSet, account_balance: float = 100.0) -> AIAnalysis:
        if isinstance(signal, TradeRejection):
            return AIAnalysis(
                decision="REJECT", ai_confidence=0.0, regime="uncertain",
                reasoning=f"Strategy rejected: {', '.join(signal.reasons)}",
                risk_factors=signal.reasons, no_trade_reasons=signal.reasons,
                raw_prompt="", raw_response="",
                model_version=settings.model_version, prompt_version=settings.prompt_version,
            )

        system = _load("system_prompt.txt")
        user = _load("market_analysis.txt").format(
            trading_mode=settings.trading_mode,
            account_balance=account_balance,
            feature_set_json=features.model_dump_json(indent=2),
            trade_signal_json=signal.model_dump_json(indent=2),
        )
        try:
            resp = self._client.messages.create(
                model=settings.model_version, max_tokens=1024,
                system=system, messages=[{"role": "user", "content": user}],
            )
            raw = resp.content[0].text
            data = json.loads(raw)
            return AIAnalysis(
                decision=data["decision"],
                ai_confidence=float(data["ai_confidence"]),
                regime=data["regime"],
                reasoning=data.get("reasoning", ""),
                risk_factors=data.get("risk_factors", []),
                no_trade_reasons=data.get("no_trade_reasons", []),
                raw_prompt=user, raw_response=raw,
                model_version=settings.model_version, prompt_version=settings.prompt_version,
            )
        except Exception as e:
            return AIAnalysis(
                decision="NO_TRADE", ai_confidence=0.0, regime="uncertain",
                reasoning=f"Failure: {e}", risk_factors=[], no_trade_reasons=[str(e)],
                raw_prompt=user if "user" in dir() else "",
                raw_response="",
                model_version=settings.model_version, prompt_version=settings.prompt_version,
                failed=True,
            )
