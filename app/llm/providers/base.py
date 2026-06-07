from abc import ABC, abstractmethod

from app.llm.schemas import LLMRequest, LLMResponse


# Common contract every backend implements. Providers translate an LLMRequest
# into their native call and return a normalized LLMResponse. They must NOT raise
# on inference failure — set LLMResponse.failed instead so the gateway can decide
# retry / cache / replay behavior.
class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        ...

    # Network providers return False: in replay mode the gateway refuses to call
    # them on a cache miss (no-live-calls invariant). A provider that can serve
    # deterministically from a fixture would override this to True.
    def supports_replay(self) -> bool:
        return False
