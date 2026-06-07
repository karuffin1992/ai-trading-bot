import logging
import time

from app.llm.providers.base import LLMProvider
from app.llm.schemas import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


# Replay/dedupe cache keyed by LLMRequest.cache_key(). In-memory by default; an
# optional session_factory lets a DB-backed layer persist responses across runs
# (the prompt_cache table is a future enhancement — TODO below). Caching makes
# replays deterministic and avoids paying for identical prompts twice.
class ReplayCache:
    def __init__(self, session_factory=None):
        self._sf = session_factory
        self._mem: dict[str, LLMResponse] = {}

    def get(self, key: str) -> LLMResponse | None:
        hit = self._mem.get(key)
        if hit is None:
            return None
        # Return a copy flagged as cached so the caller can tell.
        return hit.model_copy(update={"from_cache": True})

    def put(self, key: str, resp: LLMResponse) -> None:
        if resp.failed:
            return
        self._mem[key] = resp
        # TODO(persistence): when llm_replay_cache_enabled, also upsert into a
        # prompt_cache table via self._sf so the cache survives process restarts.


# Single entry point for all LLM inference. Dispatches an LLMRequest to the
# registered provider, applies caching, retries, timeout accounting, and enforces
# the no-live-calls-in-replay invariant.
class LLMGateway:
    def __init__(self, providers=None, session_factory=None, replay_mode=False,
                 cache=None, max_retries=2, timeout_s=30.0):
        self._providers: dict[str, LLMProvider] = {}
        self._sf = session_factory
        self.replay_mode = replay_mode
        self._cache = cache if cache is not None else ReplayCache(session_factory)
        self._max_retries = max_retries
        self._timeout_s = timeout_s
        for p in (providers or []):
            self.register(p)

    def register(self, provider: LLMProvider) -> None:
        self._providers[provider.name] = provider

    def generate(self, request: LLMRequest) -> LLMResponse:
        key = request.cache_key()

        cached = self._cache.get(key)
        if cached is not None:
            return cached

        provider = self._providers.get(request.provider)
        if provider is None:
            return LLMResponse(
                request_id=request.request_id, provider=request.provider,
                model=request.model, failed=True,
                error=f"no provider registered for '{request.provider}'",
            )

        # Replay safety: never hit a network provider on a cache miss.
        if self.replay_mode and not provider.supports_replay():
            return LLMResponse(
                request_id=request.request_id, provider=request.provider,
                model=request.model, failed=True,
                error="no-live-calls-in-replay",
            )

        resp = self._call_with_retries(provider, request)
        self._cache.put(key, resp)
        return resp

    def _call_with_retries(self, provider: LLMProvider,
                           request: LLMRequest) -> LLMResponse:
        last: LLMResponse | None = None
        for attempt in range(self._max_retries + 1):
            started = time.perf_counter()
            resp = provider.generate(request)
            if not resp.latency_ms:
                resp.latency_ms = (time.perf_counter() - started) * 1000.0
            if not resp.failed:
                logger.info("LLM %s/%s ok in %.0fms (attempt %d)",
                            request.provider, request.model, resp.latency_ms, attempt)
                return resp
            last = resp
            logger.warning("LLM %s/%s failed (attempt %d): %s",
                           request.provider, request.model, attempt, resp.error)
        return last  # type: ignore[return-value]

    # Convenience constructor wiring the default Claude backend.
    @classmethod
    def default(cls, session_factory=None, replay_mode=False) -> "LLMGateway":
        from app.llm.providers.claude import ClaudeProvider
        return cls(providers=[ClaudeProvider()], session_factory=session_factory,
                   replay_mode=replay_mode)
