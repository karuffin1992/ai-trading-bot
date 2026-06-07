import logging
from uuid import uuid4

from app.config import settings
from app.memory.schemas import MemoryEpisode, TradeReflection
from app.util.clock import now_utc

logger = logging.getLogger(__name__)


# Generates a structured post-trade reflection when a position closes, persists it
# to memory_reflections, and mirrors it into a MemoryEpisode so it becomes
# retrievable context for future analysis. `enabled` (wired from
# settings.reflection_enabled) keeps the whole thing off by default.
#
# This scaffolding fills the reflection with a deterministic template. An LLM fills
# it later (see TODO) — but NEVER during replay (no live calls), and never on the
# trade-execution path: reflection runs after the close is already recorded.
class ReflectionEngine:
    def __init__(self, session_factory=None, episodic_store=None,
                 gateway=None, enabled=True):
        self._sf = session_factory
        self._store = episodic_store
        self._gateway = gateway
        self._enabled = enabled

    def reflect_on_closed(self, positions: list[dict]) -> list[TradeReflection]:
        out: list[TradeReflection] = []
        for p in positions:
            r = self.reflect_on_position(p)
            if r is not None:
                out.append(r)
        return out

    def reflect_on_position(self, position: dict) -> TradeReflection | None:
        if not self._enabled or not self._sf:
            return None

        pnl = position.get("pnl")
        entry = position.get("entry_price")
        qty = position.get("qty") or 0.0
        outcome = _classify(pnl)
        pnl_pct = None
        if pnl is not None and entry and qty:
            notional = entry * qty
            if notional:
                pnl_pct = round(pnl / notional * 100.0, 4)

        snapshot, regime = self._feature_snapshot(position.get("trade_id"))

        reflection = TradeReflection(
            trade_id=str(position.get("trade_id", "")),
            symbol=position.get("symbol", ""),
            direction=position.get("direction", ""),
            entry_price=entry,
            exit_price=position.get("exit_price"),
            pnl=pnl,
            outcome=outcome,
            regime=regime,
            feature_snapshot=snapshot,
            feature_version=settings.strategy_version,
            strategy_version=settings.strategy_version,
            prompt_version=settings.prompt_version,
            model_version=settings.model_version,
            **_template_reflection(outcome, position, regime),
        )

        # TODO(ml): when self._gateway is set and not in replay, ask the LLM to
        # generate strengths/weaknesses/regime_notes/summary from the snapshot,
        # versioned by prompt_version, and merge over the template defaults.

        self._persist(reflection, pnl_pct, position)
        return reflection

    def _feature_snapshot(self, trade_id) -> tuple[dict, str | None]:
        if not trade_id or not self._sf:
            return {}, None
        from app.persistence.db import CycleRecord, TradeExecutionRecord
        with self._sf() as s:
            ex = s.query(TradeExecutionRecord).filter_by(
                trade_id=str(trade_id)).first()
            if ex is None or not ex.cycle_id:
                return {}, None
            cyc = s.get(CycleRecord, ex.cycle_id)
            if cyc is None:
                return {}, None
            features = cyc.features_json or {}
            ai = cyc.ai_analysis_json or {}
            return features, ai.get("regime")

    def _persist(self, reflection: TradeReflection, pnl_pct, position) -> None:
        from app.persistence.db import MemoryReflectionRecord
        with self._sf() as s:
            s.add(MemoryReflectionRecord(
                reflection_id=str(reflection.reflection_id),
                trade_id=reflection.trade_id,
                symbol=reflection.symbol,
                direction=reflection.direction,
                entry_price=reflection.entry_price,
                exit_price=reflection.exit_price,
                pnl=reflection.pnl,
                outcome=reflection.outcome,
                regime=reflection.regime,
                summary=reflection.summary,
                strengths_json=reflection.strengths,
                weaknesses_json=reflection.weaknesses,
                regime_notes_json=reflection.regime_notes,
                execution_notes_json=reflection.execution_notes,
                tags_json=reflection.tags,
                feature_snapshot_json=reflection.feature_snapshot,
                feature_version=reflection.feature_version,
                strategy_version=reflection.strategy_version,
                prompt_version=reflection.prompt_version,
                model_version=reflection.model_version,
                created_at=now_utc(),
            ))
            s.commit()

        if self._store is not None:
            self._store.write_episode(MemoryEpisode(
                kind="reflection",
                symbol=reflection.symbol,
                trade_id=reflection.trade_id,
                summary=reflection.summary,
                payload={"strengths": reflection.strengths,
                         "weaknesses": reflection.weaknesses,
                         "feature_snapshot": reflection.feature_snapshot},
                outcome=reflection.outcome,
                pnl_pct=pnl_pct,
                regime=reflection.regime,
                tags=reflection.tags,
                reflection=reflection.summary,
                feature_version=reflection.feature_version,
                strategy_version=reflection.strategy_version,
                prompt_version=reflection.prompt_version,
                model_version=reflection.model_version,
            ))


def _classify(pnl) -> str:
    if pnl is None:
        return "BREAKEVEN"
    if pnl > 0:
        return "WIN"
    if pnl < 0:
        return "LOSS"
    return "BREAKEVEN"


# Deterministic placeholder content so reflections are useful (and testable)
# before the LLM path lands. Keyed off outcome + direction.
def _template_reflection(outcome: str, position: dict, regime: str | None) -> dict:
    direction = position.get("direction", "")
    symbol = position.get("symbol", "")
    if outcome == "WIN":
        strengths = [f"{direction} {symbol} entry captured favorable move"]
        weaknesses = []
    elif outcome == "LOSS":
        strengths = []
        weaknesses = [f"{direction} {symbol} entry moved against position"]
    else:
        strengths = []
        weaknesses = []
    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "regime_notes": [f"regime at entry: {regime}"] if regime else [],
        "execution_notes": [],
        "summary": f"{outcome} on {symbol} ({direction}); pnl={position.get('pnl')}",
        "tags": [outcome.lower(), symbol] if symbol else [outcome.lower()],
    }
