from app.llm.schemas import LLMMessage, LLMRequest, LLMResponse


def _req(**kw):
    base = dict(provider="claude", model="m",
                system="sys", messages=[LLMMessage(role="user", content="hi")])
    base.update(kw)
    return LLMRequest(**base)


def test_cache_key_deterministic():
    assert _req().cache_key() == _req().cache_key()


def test_cache_key_ignores_request_id_and_metadata():
    a = _req(metadata={"a": 1})
    b = _req(metadata={"b": 2})
    assert a.request_id != b.request_id
    assert a.cache_key() == b.cache_key()


def test_cache_key_changes_with_content():
    a = _req()
    b = _req(messages=[LLMMessage(role="user", content="different")])
    assert a.cache_key() != b.cache_key()


def test_response_defaults():
    r = LLMResponse(request_id=_req().request_id, provider="claude", model="m")
    assert r.failed is False
    assert r.from_cache is False
    assert r.text == ""
