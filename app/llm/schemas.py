import hashlib
import json
from datetime import datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from app.util.clock import now_utc


# Provider-agnostic chat message. Mirrors the {role, content} shape used by every
# supported backend (Anthropic, OpenAI-compatible, Ollama) so providers only need
# to translate, never reshape.
class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


# Unified inference request. Providers consume this; nothing here is provider
# specific so a single request can be dispatched to Claude, Qwen, etc.
class LLMRequest(BaseModel):
    request_id: UUID = Field(default_factory=uuid4)
    provider: str
    model: str
    system: str = ""
    messages: list[LLMMessage] = Field(default_factory=list)
    max_tokens: int = 1024
    temperature: float = 0.0
    prompt_version: str = ""
    metadata: dict = Field(default_factory=dict)

    # Stable hash over the semantically meaningful fields (NOT request_id /
    # metadata). Used as the replay-cache key so identical prompts replay
    # deterministically. request_id and metadata are excluded on purpose.
    def cache_key(self) -> str:
        payload = {
            "provider": self.provider,
            "model": self.model,
            "system": self.system,
            "messages": [m.model_dump() for m in self.messages],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# Unified inference response. `failed=True` carries the error instead of raising
# so the gateway/caller can degrade gracefully (same pattern as AIAnalysis).
class LLMResponse(BaseModel):
    request_id: UUID
    provider: str
    model: str
    text: str = ""
    raw: dict = Field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    from_cache: bool = False
    failed: bool = False
    error: str = ""
    created_at: datetime = Field(default_factory=now_utc)
