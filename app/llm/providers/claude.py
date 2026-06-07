import time

from app.config import settings
from app.llm.providers.base import LLMProvider
from app.llm.schemas import LLMRequest, LLMResponse


# Anthropic backend. Mirrors the call AIAnalyst makes today
# (self._client.messages.create(...) -> resp.content[0].text) so behavior is
# identical when AIAnalyst is later routed through the gateway. The client is
# injectable for tests (pass a MagicMock).
class ClaudeProvider(LLMProvider):
    name = "claude"

    def __init__(self, client=None):
        if client is not None:
            self._client = client
        else:
            import anthropic
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def generate(self, request: LLMRequest) -> LLMResponse:
        user_msgs = [
            {"role": m.role, "content": m.content}
            for m in request.messages
            if m.role != "system"
        ]
        started = time.perf_counter()
        try:
            resp = self._client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens,
                system=request.system,
                messages=user_msgs,
            )
            text = resp.content[0].text
            usage = getattr(resp, "usage", None)
            return LLMResponse(
                request_id=request.request_id,
                provider=self.name,
                model=request.model,
                text=text,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
            )
        except Exception as e:
            return LLMResponse(
                request_id=request.request_id,
                provider=self.name,
                model=request.model,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                failed=True,
                error=str(e),
            )
