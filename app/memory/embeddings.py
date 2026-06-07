import hashlib
import math
import re
from abc import ABC, abstractmethod

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# Pluggable embedding backend. `embedding_version` tags every vector so retrieval
# can refuse to mix incompatible spaces (e.g. after switching models).
class EmbeddingProvider(ABC):
    name: str = "base"
    dim: int = 0
    embedding_version: str = ""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


# Default backend: dependency-free, fully deterministic feature hashing. Each
# token is hashed (blake2b) into a fixed-dim bucket with a sign, accumulated, then
# L2-normalized. Same text always yields the same vector across processes and
# machines -> replay-safe and reproducible. Quality is far below a real semantic
# model, but the interface is identical, so SentenceTransformerEmbedding can drop
# in later without touching callers.
class DeterministicHashEmbedding(EmbeddingProvider):
    name = "deterministic_hash"

    def __init__(self, dim: int = 256, embedding_version: str = "det-1.0.0"):
        self.dim = dim
        self.embedding_version = embedding_version

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _TOKEN_RE.findall((text or "").lower()):
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            n = int.from_bytes(h, "big")
            bucket = n % self.dim
            sign = 1.0 if (n >> 1) & 1 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


# Real semantic embeddings. Not the default — requires the heavy
# sentence-transformers/torch stack and is non-deterministic across versions,
# which complicates replay. Left as a stub behind the stable interface.
class SentenceTransformerEmbedding(EmbeddingProvider):
    name = "sentence_transformer"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 embedding_version: str = "st-MiniLM-L6-v2"):
        # TODO(ml): lazy-import sentence_transformers, load model, set self.dim
        # from the model's output dimension.
        raise NotImplementedError("SentenceTransformerEmbedding not yet implemented")

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError("SentenceTransformerEmbedding not yet implemented")


def make_embedder(provider: str, dim: int = 256,
                  embedding_version: str = "det-1.0.0") -> EmbeddingProvider:
    if provider == "deterministic_hash":
        return DeterministicHashEmbedding(dim=dim, embedding_version=embedding_version)
    if provider == "sentence_transformer":
        return SentenceTransformerEmbedding(embedding_version=embedding_version)
    raise ValueError(f"unknown embedding provider: {provider}")
