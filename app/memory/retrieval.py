import math
from abc import ABC, abstractmethod

from app.memory.embeddings import EmbeddingProvider
from app.memory.schemas import MemoryEpisode, RetrievedMemory


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# Storage/search backend for embedding vectors. Pluggable so a FAISS index can
# replace the SQLite brute-force scan without touching MemoryRetriever.
class VectorStore(ABC):
    @abstractmethod
    def search(self, query_vector: list[float], k: int,
               filters: dict | None = None) -> list[tuple[str, float]]:
        ...


# Default store: brute-force cosine over rows in embedding_registry. Fine for the
# thousands-of-episodes scale this system operates at. Deterministic ordering
# (score desc, then embedding_id) keeps replays reproducible.
class SQLiteVectorStore(VectorStore):
    def __init__(self, session_factory=None, embedding_version: str | None = None):
        self._sf = session_factory
        self._embedding_version = embedding_version

    def search(self, query_vector: list[float], k: int,
               filters: dict | None = None) -> list[tuple[str, float]]:
        if not self._sf:
            return []
        from app.persistence.db import EmbeddingRegistryRecord, MemoryEpisodeRecord

        symbol = (filters or {}).get("symbol")
        kind = (filters or {}).get("kind")
        scored: list[tuple[str, float]] = []
        with self._sf() as s:
            q = s.query(EmbeddingRegistryRecord)
            if self._embedding_version:
                q = q.filter(EmbeddingRegistryRecord.embedding_version
                             == self._embedding_version)
            # Resolve episode-level filters to the set of allowed embedding_ids.
            allowed: set[str] | None = None
            if symbol or kind:
                eq = s.query(MemoryEpisodeRecord.embedding_id)
                if symbol:
                    eq = eq.filter(MemoryEpisodeRecord.symbol == symbol)
                if kind:
                    eq = eq.filter(MemoryEpisodeRecord.kind == kind)
                allowed = {row[0] for row in eq.all() if row[0]}

            for row in q.all():
                if allowed is not None and row.embedding_id not in allowed:
                    continue
                score = _cosine(query_vector, row.vector_json or [])
                scored.append((row.embedding_id, score))
        # TODO(perf): replace brute force with FAISS when episode count grows.
        scored.sort(key=lambda t: (-t[1], t[0]))
        return scored[:k]


# FAISS-backed store. Optional dependency; same interface as SQLiteVectorStore.
class FaissVectorStore(VectorStore):
    def __init__(self, dim: int):
        # TODO(perf): build/load a faiss.IndexFlatIP and an id<->row mapping.
        raise NotImplementedError("FaissVectorStore is optional and not implemented")

    def search(self, query_vector, k, filters=None):
        raise NotImplementedError("FaissVectorStore is optional and not implemented")


# Ties an embedder to a vector store and hydrates hits back into MemoryEpisodes.
# This is the single retrieval entry point used by prompt augmentation and replay.
class MemoryRetriever:
    def __init__(self, embedder: EmbeddingProvider, store: VectorStore,
                 session_factory=None):
        self._embedder = embedder
        self._store = store
        self._sf = session_factory

    def retrieve(self, query_text: str, k: int = 5,
                 filters: dict | None = None) -> list[RetrievedMemory]:
        if not self._sf:
            return []
        qv = self._embedder.embed(query_text)
        hits = self._store.search(qv, k, filters)
        if not hits:
            return []

        from app.persistence.db import EmbeddingRegistryRecord, MemoryEpisodeRecord
        results: list[RetrievedMemory] = []
        with self._sf() as s:
            for rank, (embedding_id, score) in enumerate(hits):
                emb = s.get(EmbeddingRegistryRecord, embedding_id)
                if emb is None or not emb.episode_id:
                    continue
                rec = s.get(MemoryEpisodeRecord, emb.episode_id)
                if rec is None:
                    continue
                results.append(RetrievedMemory(
                    episode=_record_to_episode(rec), score=score, rank=rank))
        return results


def _record_to_episode(rec) -> MemoryEpisode:
    return MemoryEpisode(
        episode_id=rec.episode_id,
        kind=rec.kind,
        symbol=rec.symbol,
        cycle_id=rec.cycle_id,
        trade_id=rec.trade_id,
        summary=rec.summary or "",
        payload=rec.payload_json or {},
        outcome=rec.outcome,
        pnl_pct=rec.pnl_pct,
        holding_time_minutes=rec.holding_time_minutes,
        regime=rec.regime,
        tags=rec.tags_json or [],
        reflection=rec.reflection,
        embedding_id=rec.embedding_id,
        feature_version=rec.feature_version or "",
        strategy_version=rec.strategy_version or "",
        prompt_version=rec.prompt_version or "",
        model_version=rec.model_version or "",
        created_at=rec.created_at,
    )
