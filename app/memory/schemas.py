from datetime import datetime
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from app.util.clock import now_utc

EpisodeKind = Literal["trade", "cycle", "reflection", "regime_note"]
Outcome = Literal["WIN", "LOSS", "BREAKEVEN", "NO_TRADE"]


# A single unit of memory. `summary` is the human/LLM-readable text that gets
# embedded for retrieval and injected into prompts; `payload` holds the structured
# detail. Episodic (trade/cycle), semantic (regime_note), and procedural memories
# all share this shape, distinguished by `kind` + `tags`.
class MemoryEpisode(BaseModel):
    episode_id: UUID = Field(default_factory=uuid4)
    kind: EpisodeKind
    symbol: str
    cycle_id: Optional[str] = None
    trade_id: Optional[str] = None

    summary: str
    payload: dict = Field(default_factory=dict)

    outcome: Optional[Outcome] = None
    pnl_pct: Optional[float] = None
    holding_time_minutes: Optional[int] = None
    regime: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    reflection: Optional[str] = None

    embedding_id: Optional[str] = None

    # Provenance — which versions produced this memory, so retrieval can filter
    # to compatible runs and replays stay reproducible.
    feature_version: str = ""
    strategy_version: str = ""
    prompt_version: str = ""
    model_version: str = ""

    created_at: datetime = Field(default_factory=now_utc)


# A scored hit from the retriever. `score` is cosine similarity; `rank` is 0-based
# position in the top-k. The convenience fields mirror the most-used episode
# attributes so prompt builders don't have to reach into `episode`.
class RetrievedMemory(BaseModel):
    episode: MemoryEpisode
    score: float
    rank: int

    @property
    def summary(self) -> str:
        return self.episode.summary

    @property
    def outcome(self) -> Optional[str]:
        return self.episode.outcome

    @property
    def pnl_pct(self) -> Optional[float]:
        return self.episode.pnl_pct


# Structured post-trade reflection. In this scaffolding pass the lists/summary are
# filled by a deterministic template; an LLM fills them later (see ReflectionEngine
# TODO). Stored separately from the raw trade, then mirrored into a MemoryEpisode.
class TradeReflection(BaseModel):
    reflection_id: UUID = Field(default_factory=uuid4)
    trade_id: str

    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    regime_notes: list[str] = Field(default_factory=list)
    execution_notes: list[str] = Field(default_factory=list)
    summary: str = ""

    symbol: str = ""
    direction: str = ""
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    outcome: Optional[Outcome] = None
    regime: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    feature_snapshot: dict = Field(default_factory=dict)

    feature_version: str = ""
    strategy_version: str = ""
    prompt_version: str = ""
    model_version: str = ""

    created_at: datetime = Field(default_factory=now_utc)
