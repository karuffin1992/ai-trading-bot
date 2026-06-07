from app.llm.gateway import LLMGateway, ReplayCache
from app.llm.providers.base import LLMProvider
from app.llm.schemas import LLMMessage, LLMRequest, LLMResponse


class FakeProvider(LLMProvider):
    name = "fake"

    def __init__(self, replay=False, text="ok"):
        self.calls = 0
        self._replay = replay
        self._text = text

    def generate(self, request):
        self.calls += 1
        return LLMResponse(request_id=request.request_id, provider=self.name,
                           model=request.model, text=self._text)

    def supports_replay(self):
        return self._replay


class FlakyProvider(LLMProvider):
    name = "fake"

    def __init__(self, fail_times):
        self.calls = 0
        self._fail_times = fail_times

    def generate(self, request):
        self.calls += 1
        if self.calls <= self._fail_times:
            return LLMResponse(request_id=request.request_id, provider=self.name,
                               model=request.model, failed=True, error="transient")
        return LLMResponse(request_id=request.request_id, provider=self.name,
                           model=request.model, text="recovered")


def _req(provider="fake"):
    return LLMRequest(provider=provider, model="m",
                      messages=[LLMMessage(role="user", content="hi")])


def test_cache_hit_skips_provider():
    p = FakeProvider()
    gw = LLMGateway(providers=[p])
    r1 = gw.generate(_req())
    r2 = gw.generate(_req())
    assert r1.from_cache is False
    assert r2.from_cache is True
    assert p.calls == 1


def test_replay_blocks_live_provider_on_miss():
    p = FakeProvider(replay=False)
    gw = LLMGateway(providers=[p], replay_mode=True)
    r = gw.generate(_req())
    assert r.failed is True
    assert r.error == "no-live-calls-in-replay"
    assert p.calls == 0


def test_replay_allows_replay_capable_provider():
    p = FakeProvider(replay=True)
    gw = LLMGateway(providers=[p], replay_mode=True)
    r = gw.generate(_req())
    assert r.failed is False
    assert p.calls == 1


def test_unknown_provider_fails_cleanly():
    gw = LLMGateway(providers=[FakeProvider()])
    r = gw.generate(_req(provider="missing"))
    assert r.failed is True
    assert "no provider" in r.error


def test_retry_recovers():
    p = FlakyProvider(fail_times=1)
    gw = LLMGateway(providers=[p], max_retries=2)
    r = gw.generate(_req())
    assert r.failed is False
    assert r.text == "recovered"
    assert p.calls == 2


def test_failed_response_not_cached():
    cache = ReplayCache()
    cache.put("k", LLMResponse(request_id=_req().request_id, provider="fake",
                               model="m", failed=True, error="x"))
    assert cache.get("k") is None
