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
    # Class-level defaults so instances built via AIAnalyst.__new__(...) in tests
    # (which skip __init__) still resolve these attributes.
    _gateway = None
    _retriever = None
    _summarizer = None

    def __init__(self, gateway=None, retriever=None, summarizer=None):
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._gateway = gateway
        self._retriever = retriever
        self._summarizer = summarizer

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
        ctx = self.build_prompt_context(signal, features, account_balance)
        user = self.build_prompt(ctx)
        try:
            raw = self._call_model(system, user)
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

    # Pure context assembly for the prompt — no client, no network. Used by
    # analyze() and by the golden replay runner.
    def build_prompt_context(self, signal: TradeSignal, features: FeatureSet,
                             account_balance: float = 100.0) -> dict:
        return {
            "trading_mode": settings.trading_mode,
            "account_balance": account_balance,
            "feature_set_json": features.model_dump_json(indent=2),
            "trade_signal_json": signal.model_dump_json(indent=2),
            "memory_block": self._memory_block(signal, features),
        }

    # Formats the template from a context dict. Memory block concatenated under
    # its header only when present, so the disabled path is byte-identical to legacy.
    def build_prompt(self, ctx: dict) -> str:
        user = _load("market_analysis.txt").format(
            trading_mode=ctx["trading_mode"],
            account_balance=ctx["account_balance"],
            feature_set_json=ctx["feature_set_json"],
            trade_signal_json=ctx["trade_signal_json"],
        )
        if ctx.get("memory_block"):
            user = f"{user}\n\nRELEVANT PAST EPISODES:\n{ctx['memory_block']}"
        return user

    # Routes through the LLM gateway when one is injected (provider-agnostic,
    # replay-cached); otherwise uses the direct Anthropic client (legacy path,
    # preserves the self._client test seam).
    def _call_model(self, system: str, user: str) -> str:
        if self._gateway is not None:
            from app.llm.schemas import LLMMessage, LLMRequest
            req = LLMRequest(
                provider=settings.llm_provider, model=settings.model_version,
                system=system, messages=[LLMMessage(role="user", content=user)],
                max_tokens=1024, prompt_version=settings.prompt_version,
            )
            resp = self._gateway.generate(req)
            if resp.failed:
                raise RuntimeError(resp.error or "gateway inference failed")
            return resp.text
        out = self._client.messages.create(
            model=settings.model_version, max_tokens=1024,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return out.content[0].text

    # Builds the injected memory text. Returns "" unless injection is enabled and a
    # retriever is wired — so the disabled path adds nothing to the prompt.
    def _memory_block(self, signal: TradeSignal, features: FeatureSet) -> str:
        if not settings.memory_injection_enabled or self._retriever is None:
            return ""
        query = (f"{features.symbol} regime price={features.price} "
                 f"rsi={features.rsi} vix={features.vix} "
                 f"{signal.direction} {signal.strategy}")
        hits = self._retriever.retrieve(query, k=settings.memory_retrieval_k)
        if not hits:
            return ""
        if self._summarizer is not None:
            return self._summarizer.summarize_episodes(
                hits, settings.memory_token_budget)
        # Fallback rendering when no summarizer is injected.
        char_budget = settings.memory_token_budget * 4
        lines, used = [], 0
        for i, rm in enumerate(hits, start=1):
            line = f"{i}. {rm.outcome or 'NO_TRADE'} | {rm.summary}"
            if used + len(line) + 1 > char_budget:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)
