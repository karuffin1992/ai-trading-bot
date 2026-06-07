import pytest
from sqlalchemy import create_engine

from app.memory.embeddings import DeterministicHashEmbedding
from app.memory.episodic_store import EpisodicStore
from app.memory.memory_compactor import MemoryCompactor
from app.memory.schemas import MemoryEpisode
from app.persistence.db import Base, get_session


@pytest.fixture
def sf():
    e = create_engine("sqlite:///data/test_compactor.db")
    Base.metadata.create_all(e)
    yield lambda: get_session(e)
    Base.metadata.drop_all(e)


def _seed(sf, n):
    store = EpisodicStore(sf, embedder=DeterministicHashEmbedding(dim=16))
    for i in range(n):
        store.write_episode(MemoryEpisode(kind="trade", symbol="SPY",
                                          summary=f"episode {i}"))


def test_compact_drops_oldest_beyond_cap(sf):
    _seed(sf, 5)
    result = MemoryCompactor(sf, max_episodes=3).compact()
    assert result["removed"] == 2
    assert result["kept"] == 3
    from app.persistence.db import EmbeddingRegistryRecord, MemoryEpisodeRecord
    with sf() as s:
        assert s.query(MemoryEpisodeRecord).count() == 3
        # Embeddings for dropped episodes are gone too.
        assert s.query(EmbeddingRegistryRecord).count() == 3


def test_compact_noop_under_cap(sf):
    _seed(sf, 2)
    result = MemoryCompactor(sf, max_episodes=10).compact()
    assert result["removed"] == 0
    assert result["kept"] == 2


def test_prune_orphans(sf):
    _seed(sf, 2)
    from app.persistence.db import EmbeddingRegistryRecord, MemoryEpisodeRecord
    with sf() as s:
        s.query(MemoryEpisodeRecord).delete()
        s.commit()
    pruned = MemoryCompactor(sf).prune_orphans()
    assert pruned == 2
    with sf() as s:
        assert s.query(EmbeddingRegistryRecord).count() == 0


def test_no_session_factory_noop():
    c = MemoryCompactor(None)
    assert c.compact() == {"removed": 0, "merged": 0, "kept": 0}
    assert c.prune_orphans() == 0
