import json
from unittest.mock import MagicMock
from uuid import uuid4

from app.util.clock import now_utc
from app.ai import analyst as analyst_mod
from app.ai.analyst import AIAnalyst
from app.models.ai import AIAnalysis
from app.models.market import FeatureSet
from app.models.signals import TradeSignal


def make_fs():
    return FeatureSet(symbol="SPY", timeframe="1D", timestamp=now_utc(),
                      feature_version="1.0.0", price=521.0, vwap=519.0,
                      ema_9=521.5, ema_20=519.0, ema_50=515.0, rsi=55.0,
                      macd=0.5, macd_signal=0.3, macd_hist=0.2, atr=2.1,
                      vix=18.0, relative_volume=1.5, spread_proxy=0.0003,
                      news_sentiment=0.1)


def make_signal():
    return TradeSignal(trade_id=uuid4(), symbol="SPY",
                       strategy="spy_trend_following", direction="long",
                       strategy_confidence=0.74, entry_price=521.0,
                       stop_loss=516.8, take_profit=527.3)


def _approve_response():
    return {"decision": "APPROVE", "ai_confidence": 0.8, "regime": "bullish",
            "reasoning": "ok", "risk_factors": [], "no_trade_reasons": []}


def _mock_client(response_dict):
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps(response_dict))]
    client = MagicMock()
    client.messages.create.return_value = msg
    return client


class _FakeHit:
    def __init__(self, summary, outcome="WIN", pnl_pct=1.2):
        self.summary = summary
        self.outcome = outcome
        self.pnl_pct = pnl_pct


class _FakeRetriever:
    def __init__(self, hits):
        self._hits = hits
        self.calls = []

    def retrieve(self, query, k=5):
        self.calls.append((query, k))
        return self._hits


def test_disabled_prompt_has_no_memory_header(monkeypatch):
    monkeypatch.setattr(analyst_mod.settings, "memory_injection_enabled", False)
    a = AIAnalyst.__new__(AIAnalyst)
    a._client = _mock_client(_approve_response())
    a._retriever = _FakeRetriever([_FakeHit("past trade")])
    r = a.analyze(make_signal(), make_fs())
    assert "RELEVANT PAST EPISODES:" not in r.raw_prompt


def test_disabled_prompt_byte_identical_to_no_retriever(monkeypatch):
    monkeypatch.setattr(analyst_mod.settings, "memory_injection_enabled", False)
    a1 = AIAnalyst.__new__(AIAnalyst)
    a1._client = _mock_client(_approve_response())
    a1._retriever = _FakeRetriever([_FakeHit("past trade")])
    sig, fs = make_signal(), make_fs()
    p_with = a1.analyze(sig, fs).raw_prompt

    a2 = AIAnalyst.__new__(AIAnalyst)
    a2._client = _mock_client(_approve_response())
    a2._retriever = None
    p_without = a2.analyze(sig, fs).raw_prompt
    assert p_with == p_without


def test_enabled_injects_header_within_budget(monkeypatch):
    monkeypatch.setattr(analyst_mod.settings, "memory_injection_enabled", True)
    monkeypatch.setattr(analyst_mod.settings, "memory_retrieval_k", 5)
    monkeypatch.setattr(analyst_mod.settings, "memory_token_budget", 50)
    hits = [_FakeHit(f"episode number {i} with some descriptive text") for i in range(20)]
    retriever = _FakeRetriever(hits)
    a = AIAnalyst.__new__(AIAnalyst)
    a._client = _mock_client(_approve_response())
    a._retriever = retriever
    a._summarizer = None
    r = a.analyze(make_signal(), make_fs())
    assert "RELEVANT PAST EPISODES:" in r.raw_prompt
    assert retriever.calls and retriever.calls[0][1] == 5
    block = r.raw_prompt.split("RELEVANT PAST EPISODES:\n", 1)[1]
    assert len(block) <= 50 * 4 + 10


def test_enabled_no_hits_no_header(monkeypatch):
    monkeypatch.setattr(analyst_mod.settings, "memory_injection_enabled", True)
    a = AIAnalyst.__new__(AIAnalyst)
    a._client = _mock_client(_approve_response())
    a._retriever = _FakeRetriever([])
    r = a.analyze(make_signal(), make_fs())
    assert "RELEVANT PAST EPISODES:" not in r.raw_prompt


class _FakeGateway:
    def __init__(self, text):
        self._text = text
        self.requests = []

    def generate(self, req):
        self.requests.append(req)
        resp = MagicMock()
        resp.failed = False
        resp.text = self._text
        resp.error = ""
        return resp


def test_gateway_path_returns_parsed_analysis(monkeypatch):
    monkeypatch.setattr(analyst_mod.settings, "memory_injection_enabled", False)
    gw = _FakeGateway(json.dumps(_approve_response()))
    a = AIAnalyst.__new__(AIAnalyst)
    a._client = MagicMock()
    a._gateway = gw
    r = a.analyze(make_signal(), make_fs())
    assert isinstance(r, AIAnalysis)
    assert r.decision == "APPROVE"
    assert r.failed is False
    a._client.messages.create.assert_not_called()
    assert len(gw.requests) == 1
