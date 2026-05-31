import pytest
from unittest.mock import patch, MagicMock
from uuid import uuid4
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    with patch("app.scheduler.create_scheduler") as m:
        m.return_value = MagicMock()
        import importlib, main as m2
        importlib.reload(m2)
        yield TestClient(m2.app)

def test_health(client):
    assert client.get("/health").status_code == 200

def test_approve_missing_trade(client):
    with patch("main._get_pending_trade", return_value=None):
        r = client.post(f"/approve/{uuid4()}")
        assert r.status_code == 404

def test_approve_executes(client):
    tid = uuid4()
    sig = {"type":"SIGNAL","trade_id":str(tid),"symbol":"SPY",
           "strategy":"spy_trend_following","direction":"long",
           "strategy_confidence":0.74,"entry_price":521.0,
           "stop_loss":516.8,"take_profit":527.3,"position_side":"long"}
    with patch("main._get_pending_trade", return_value=sig), \
         patch("main._execute_approved_trade", return_value={"status":"executed"}):
        r = client.post(f"/approve/{tid}")
        assert r.status_code == 200
