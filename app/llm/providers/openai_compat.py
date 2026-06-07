from app.llm.providers.base import LLMProvider
from app.llm.schemas import LLMRequest, LLMResponse


# Generic OpenAI-compatible backend (LM Studio, vLLM, llama.cpp server, etc.).
# Stub: targets the /v1/chat/completions contract so any local server speaking it
# works without a bespoke provider.
class OpenAICompatProvider(LLMProvider):
    name = "openai_compat"

    def __init__(self, base_url: str, api_key: str = "", model: str = ""):
        self._base_url = base_url
        self._api_key = api_key
        self._model = model

    def generate(self, request: LLMRequest) -> LLMResponse:
        # TODO(provider): POST to {base_url}/v1/chat/completions with system+user
        # messages, read choices[0].message.content -> LLMResponse.text and
        # usage.{prompt,completion}_tokens.
        raise NotImplementedError("OpenAICompatProvider not yet implemented")
