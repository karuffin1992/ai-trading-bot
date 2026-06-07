from app.llm.providers.base import LLMProvider
from app.llm.schemas import LLMRequest, LLMResponse


# Self-hosted Qwen backend (e.g. served via vLLM / TGI). Stub: the interface is
# fixed so the gateway can register it unchanged once implemented.
class QwenProvider(LLMProvider):
    name = "qwen"

    def __init__(self, base_url: str = "", api_key: str = "", model: str = ""):
        self._base_url = base_url
        self._api_key = api_key
        self._model = model

    def generate(self, request: LLMRequest) -> LLMResponse:
        # TODO(provider): POST to the Qwen inference server, parse the completion
        # into LLMResponse. Use httpx (already a dependency) with request.timeout.
        raise NotImplementedError("QwenProvider not yet implemented")
