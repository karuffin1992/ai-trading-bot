from app.memory.schemas import RetrievedMemory, TradeReflection


# Produces the short text used for embedding and prompt injection. Deterministic
# template stubs for now; an LLM-backed map-reduce summarizer drops in later
# behind the same methods (see TODOs). `gateway` is accepted now so callers can be
# wired without a later signature change.
class Summarizer:
    def __init__(self, gateway=None, max_chars: int = 400):
        self._gateway = gateway
        self._max_chars = max_chars

    def summarize_cycle(self, ctx) -> str:
        # TODO(ml): LLM summary of the cycle. Stub = compact deterministic line.
        parts = []
        if getattr(ctx, "features", None) is not None:
            f = ctx.features
            parts.append(f"{f.symbol} price={f.price} rsi={f.rsi} vix={f.vix}")
        if getattr(ctx, "ai_analysis", None) is not None:
            parts.append(f"decision={ctx.ai_analysis.decision} "
                         f"regime={ctx.ai_analysis.regime}")
        return _truncate("; ".join(parts), self._max_chars)

    def summarize_reflection(self, reflection: TradeReflection) -> str:
        # TODO(ml): richer LLM lesson. Stub = the templated summary.
        return _truncate(reflection.summary, self._max_chars)

    # Renders retrieved memories into a token-budgeted block for prompt injection.
    # token_budget is approximated as ~4 chars/token. Truncates at episode
    # boundaries so partial lines never leak into the prompt.
    def summarize_episodes(self, episodes: list[RetrievedMemory],
                           token_budget: int) -> str:
        char_budget = token_budget * 4
        lines: list[str] = []
        used = 0
        for i, rm in enumerate(episodes, start=1):
            ep = rm.episode
            date = ep.created_at.date().isoformat() if ep.created_at else "?"
            outcome = ep.outcome or "NO_TRADE"
            pnl = f" ({ep.pnl_pct:+.1f}%)" if ep.pnl_pct is not None else ""
            line = f"{i}. {date} | {outcome}{pnl} | {ep.summary}"
            if used + len(line) + 1 > char_budget:
                break
            lines.append(line)
            used += len(line) + 1
        # TODO(ml): LLM map-reduce when episodes exceed the budget.
        return "\n".join(lines)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"
