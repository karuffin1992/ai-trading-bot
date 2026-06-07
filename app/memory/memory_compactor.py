import logging

logger = logging.getLogger(__name__)


# Offline maintenance for the memory store. NEVER runs on the trade path — invoked
# only by a scheduled/manual job. Keeps the episode table bounded and the
# embedding registry consistent.
class MemoryCompactor:
    def __init__(self, session_factory=None, retriever=None,
                 max_episodes: int = 5000, dedupe_threshold: float = 0.98):
        self._sf = session_factory
        self._retriever = retriever
        self._max_episodes = max_episodes
        self._dedupe_threshold = dedupe_threshold

    # Drops the oldest episodes beyond max_episodes (plus their embeddings).
    # Returns counts so a caller/report can audit what changed.
    def compact(self) -> dict:
        if not self._sf:
            return {"removed": 0, "merged": 0, "kept": 0}
        from app.persistence.db import EmbeddingRegistryRecord, MemoryEpisodeRecord
        removed = 0
        with self._sf() as s:
            total = s.query(MemoryEpisodeRecord).count()
            overflow = max(0, total - self._max_episodes)
            if overflow:
                old = (s.query(MemoryEpisodeRecord)
                       .order_by(MemoryEpisodeRecord.created_at.asc())
                       .limit(overflow).all())
                for rec in old:
                    if rec.embedding_id:
                        emb = s.get(EmbeddingRegistryRecord, rec.embedding_id)
                        if emb is not None:
                            s.delete(emb)
                    s.delete(rec)
                    removed += 1
            s.commit()
            kept = s.query(MemoryEpisodeRecord).count()
        # TODO(ml): cluster near-duplicate episodes (cosine >= dedupe_threshold)
        # and merge them into a single semantic episode instead of just dropping.
        return {"removed": removed, "merged": 0, "kept": kept}

    # Deletes embedding rows whose episode no longer exists.
    def prune_orphans(self) -> int:
        if not self._sf:
            return 0
        from app.persistence.db import EmbeddingRegistryRecord, MemoryEpisodeRecord
        pruned = 0
        with self._sf() as s:
            live = {row[0] for row in s.query(MemoryEpisodeRecord.episode_id).all()}
            for emb in s.query(EmbeddingRegistryRecord).all():
                if emb.episode_id not in live:
                    s.delete(emb)
                    pruned += 1
            s.commit()
        return pruned
