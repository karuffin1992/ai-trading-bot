import pytest
from datetime import date
from uuid import uuid4
from app.util.clock import now_utc
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.persistence.db import Base, CycleRecord
from app.persistence.logger import PipelineLogger
from app.pipeline.context import PipelineContext
from app.models.market import MarketData
from app.models.signals import TradeRejection

TEST_DB = "sqlite:///data/test_logger.db"

@pytest.fixture
def logger():
    engine = create_engine(TEST_DB)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    @contextmanager
    def factory():
        s = Session(); yield s; s.close()
    yield PipelineLogger(session_factory=factory)
    Base.metadata.drop_all(engine)

def make_ctx():
    ctx = PipelineContext()
    ctx.market_data = MarketData(
        symbol="SPY", timestamp=now_utc(), pipeline_version="1.0.0",
        open=520.0, high=522.0, low=519.0, close=521.0, volume=1e6,
        bid=521.0, ask=521.1, spread_proxy=0.0001, vix=18.5, news_sentiment=0.1)
    ctx.signal = TradeRejection(symbol="SPY", strategy="spy_trend_following",
                                reasons=["test"], strategy_confidence=0.3)
    return ctx

def test_write_cycle_persists(logger):
    ctx = make_ctx()
    logger.write_cycle(ctx)
    from sqlalchemy import create_engine
    e2 = create_engine(TEST_DB)
    Session2 = sessionmaker(bind=e2)
    with Session2() as s:
        rec = s.get(CycleRecord, str(ctx.cycle_id))
        assert rec is not None

def test_report_file_created(logger, tmp_path, monkeypatch):
    monkeypatch.setattr("app.persistence.logger.REPORTS_DIR", str(tmp_path))
    ctx = make_ctx()
    logger.write_cycle(ctx)
    logger.write_report(ctx, report_date=date(2026, 5, 16))
    assert (tmp_path / "2026-05-16.md").exists()
