from unittest.mock import MagicMock

from app.llm.providers.claude import ClaudeProvider
from app.llm.schemas import LLMMessage, LLMRequest


def _provider(text="hello"):
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    msg.usage = MagicMock(input_tokens=11, output_tokens=7)
    client = MagicMock()
    client.messages.create.return_value = msg
    return ClaudeProvider(client=client), client


def _req():
    return LLMRequest(provider="claude", model="claude-x", system="sys",
                      messages=[LLMMessage(role="user", content="hi")])


def test_generate_parses_text_and_usage():
    p, client = _provider("answer")
    resp = p.generate(_req())
    assert resp.failed is False
    assert resp.text == "answer"
    assert resp.input_tokens == 11
    assert resp.output_tokens == 7
    # System goes through the system kwarg, not the messages list.
    _, kwargs = client.messages.create.call_args
    assert kwargs["system"] == "sys"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_generate_failure_sets_failed():
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("boom")
    p = ClaudeProvider(client=client)
    resp = p.generate(_req())
    assert resp.failed is True
    assert "boom" in resp.error


def test_does_not_support_replay():
    p, _ = _provider()
    assert p.supports_replay() is False
