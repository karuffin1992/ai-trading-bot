from app.llm.providers.base import LLMProvider
from app.llm.schemas import LLMRequest, LLMResponse


# Local Ollama backend (http://localhost:11434). Stub: hosts Qwen and other
# local models via the Ollama /api/chat endpoint.
class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434"):
        self._base_url = base_url

    def generate(self, request: LLMRequest) -> LLMResponse:
        # TODO(provider): POST {model, messages, options:{temperature,num_predict}}
        # to {base_url}/api/chat, map message.content -> LLMResponse.text.
        raise NotImplementedError("OllamaProvider not yet implemented")
