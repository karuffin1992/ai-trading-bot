import hashlib
from uuid import uuid4

from app.memory.embeddings import EmbeddingProvider
from app.memory.schemas import MemoryEpisode
from app.util.clock import now_utc


# Sole writer of memory_episodes + embedding_registry. Persisting an episode
# embeds its summary, records the vector, and links the two by embedding_id so the
# retriever can hydrate hits. No-op safe when constructed without a session
# factory (mirrors the DI pattern used across the codebase).
class EpisodicStore:
    def __init__(self, session_factory=None, embedder: EmbeddingProvider | None = None):
        self._sf = session_factory
        self._embedder = embedder

    def write_episode(self, episode: MemoryEpisode) -> str | None:
        if not self._sf:
            return None
        from app.persistence.db import EmbeddingRegistryRecord, MemoryEpisodeRecord

        embedding_id: str | None = None
        with self._sf() as s:
            if self._embedder is not None and episode.summary:
                vec = self._embedder.embed(episode.summary)
                embedding_id = str(uuid4())
                episode.embedding_id = embedding_id
                s.add(EmbeddingRegistryRecord(
                    embedding_id=embedding_id,
                    episode_id=str(episode.episode_id),
                    provider=self._embedder.name,
                    dim=self._embedder.dim,
                    vector_json=vec,
                    content_hash=hashlib.sha256(
                        episode.summary.encode("utf-8")).hexdigest(),
                    embedding_version=self._embedder.embedding_version,
                    created_at=now_utc(),
                ))
            s.add(_episode_to_record(episode))
            s.commit()
        return str(episode.episode_id)

    def get_episode(self, episode_id: str) -> MemoryEpisode | None:
        if not self._sf:
            return None
        from app.memory.retrieval import _record_to_episode
        from app.persistence.db import MemoryEpisodeRecord
        with self._sf() as s:
            rec = s.get(MemoryEpisodeRecord, str(episode_id))
            return _record_to_episode(rec) if rec else None

    def recent(self, symbol: str, kind: str | None = None,
               limit: int = 50) -> list[MemoryEpisode]:
        if not self._sf:
            return []
        from app.memory.retrieval import _record_to_episode
        from app.persistence.db import MemoryEpisodeRecord
        with self._sf() as s:
            q = s.query(MemoryEpisodeRecord).filter_by(symbol=symbol)
            if kind:
                q = q.filter_by(kind=kind)
            rows = q.order_by(MemoryEpisodeRecord.created_at.desc()).limit(limit).all()
            return [_record_to_episode(r) for r in rows]


def _episode_to_record(e: MemoryEpisode):
    from app.persistence.db import MemoryEpisodeRecord
    return MemoryEpisodeRecord(
        episode_id=str(e.episode_id),
        kind=e.kind,
        symbol=e.symbol,
        cycle_id=e.cycle_id,
        trade_id=e.trade_id,
        summary=e.summary,
        payload_json=e.payload,
        outcome=e.outcome,
        pnl_pct=e.pnl_pct,
        holding_time_minutes=e.holding_time_minutes,
        regime=e.regime,
        tags_json=e.tags,
        reflection=e.reflection,
        embedding_id=e.embedding_id,
        feature_version=e.feature_version,
        strategy_version=e.strategy_version,
        prompt_version=e.prompt_version,
        model_version=e.model_version,
        created_at=e.created_at or now_utc(),
    )
