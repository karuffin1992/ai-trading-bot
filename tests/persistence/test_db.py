import pytest
from sqlalchemy import create_engine, inspect
from app.persistence.db import Base, get_session, CycleRecord, PendingTradeRecord, KillSwitchRecord

TEST_DB = "sqlite:///data/test_schema.db"

@pytest.fixture
def engine():
    e = create_engine(TEST_DB)
    Base.metadata.create_all(e)
    yield e
    Base.metadata.drop_all(e)

def test_required_tables_exist(engine):
    tables = inspect(engine).get_table_names()
    for t in ["cycles","trade_executions","positions","daily_summary",
               "pending_trades","kill_switch_state"]:
        assert t in tables, f"Missing table: {t}"

def test_kill_switch_singleton(engine):
    with get_session(engine) as s:
        s.add(KillSwitchRecord(id=1, active=False, reason=""))
        s.commit()
        rec = s.get(KillSwitchRecord, 1)
        assert rec.active is False

def test_pending_trade_status(engine):
    from uuid import uuid4
    from datetime import datetime
    with get_session(engine) as s:
        s.add(PendingTradeRecord(
            id=str(uuid4()), cycle_id=str(uuid4()),
            signal_json={}, status="PENDING_APPROVAL",
            created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
        ))
        s.commit()
        rec = s.query(PendingTradeRecord).first()
        assert rec.status == "PENDING_APPROVAL"
