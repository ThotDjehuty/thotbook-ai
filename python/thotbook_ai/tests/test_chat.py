"""Tests for ChatClient (mock transport) + optional live integration.

The mock tests never touch the network. The live integration tests are
skipped automatically when llm-serve is not running on 127.0.0.1:8100.
"""

from __future__ import annotations

import pytest
import requests

from thotbook_ai.chat import ChatClient, ChatMessage, DEFAULT_BASE_URL, DEFAULT_MODEL


class FakeResponse:
    def __init__(self, status=200, json_data=None):
        self.status_code = status
        self._json = json_data or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """Records GET/POST URLs and returns canned payloads."""

    def __init__(self, replies, http_error_on=None):
        self.replies = replies
        self.http_error_on = http_error_on or []
        self.calls = []

    def get(self, url, timeout=0):
        self.calls.append(("GET", url))
        if url in self.http_error_on:
            raise requests.ConnectionError("refused")
        return self.replies.get(url, FakeResponse(json_data={}))

    def post(self, url, json=None, timeout=0):
        self.calls.append(("POST", url, json))
        if url in self.http_error_on:
            raise requests.ConnectionError("refused")
        return self.replies.get(url, FakeResponse(json_data={}))


def make_client(session):
    return ChatClient(session=session)


def test_defaults_sane():
    client = ChatClient()  # no network used at construction
    assert client.base_url == DEFAULT_BASE_URL
    assert client.root_url == "http://127.0.0.1:8100"
    assert client.model == DEFAULT_MODEL


def test_is_online_true_when_health_ok():
    ok = FakeResponse(json_data={"status": "ok"})
    # build session replies keyed on the real root health url
    client = ChatClient(base_url=DEFAULT_BASE_URL, session=FakeSession({
        "http://127.0.0.1:8100/health": ok,
    }))
    assert client.is_online() is True
    assert client.health() == {"status": "ok"}


def test_is_online_false_on_error():
    client = ChatClient(base_url=DEFAULT_BASE_URL, session=FakeSession({}, http_error_on=["http://127.0.0.1:8100/health"]))
    assert client.is_online() is False


def test_models_lists_ids():
    payload = FakeResponse(json_data={
        "object": "list",
        "data": [{"id": "Qwen/Qwen3-8B", "object": "model", "owned_by": "thotbook"}],
    })
    client = ChatClient(session=FakeSession({"http://127.0.0.1:8100/v1/models": payload}))
    assert client.models() == ["Qwen/Qwen3-8B"]


def test_complete_parses_choice_and_posts_correct_payload():
    payload = FakeResponse(json_data={
        "object": "chat.completion",
        "model": "Qwen/Qwen3-8B",
        "created": 123456,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "42"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 7, "completion_tokens": 1},
    })
    fake = FakeSession({"http://127.0.0.1:8100/v1/chat/completions": payload})
    client = ChatClient(session=fake)

    resp = client.complete([ChatMessage(role="user", content="what is 6*7?")], max_tokens=16, temperature=0.7, top_p=0.9)

    assert resp.text == "42"
    assert resp.model == "Qwen/Qwen3-8B"
    assert resp.prompt_tokens == 7
    assert resp.completion_tokens == 1
    assert resp.created == 123456
    assert resp.raw["choices"][0]["message"]["content"] == "42"

    method, url, body = fake.calls[-1]
    assert method == "POST"
    assert url == "http://127.0.0.1:8100/v1/chat/completions"
    assert body["model"] == "Qwen/Qwen3-8B"
    assert body["messages"] == [{"role": "user", "content": "what is 6*7?"}]
    assert body["max_tokens"] == 16
    assert body["temperature"] == 0.7
    assert body["top_p"] == 0.9


def test_ask_injects_optional_system_message():
    payload = FakeResponse(json_data={
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
    })
    fake = FakeSession({"http://127.0.0.1:8100/v1/chat/completions": payload})
    client = ChatClient(session=fake)

    client.ask("hi", system="strict")

    _, _, body = fake.calls[-1]
    assert body["messages"] == [
        {"role": "system", "content": "strict"},
        {"role": "user", "content": "hi"},
    ]


def test_complete_rejects_empty_messages():
    client = ChatClient(session=FakeSession({}))
    with pytest.raises(ValueError, match="at least one message"):
        client.complete([])


def test_ask_http_error_raises():
    client = ChatClient(session=FakeSession({}, http_error_on=["http://127.0.0.1:8100/v1/chat/completions"]))
    with pytest.raises(requests.RequestException):
        client.ask("hello")


def test_health_uses_root_url_not_base():
    payload = FakeResponse(json_data={"status": "ok"})
    fake = FakeSession({"http://127.0.0.1:8100/health": payload})
    client = ChatClient(session=fake)
    assert client.health() == {"status": "ok"}
    assert fake.calls[0] == ("GET", "http://127.0.0.1:8100/health")


# ---------------------------------------------------------------------------
# LIVE integration — skipped unless llm-serve is actually running.
# ---------------------------------------------------------------------------

def _llm_up() -> bool:
    try:
        return ChatClient().is_online()
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _llm_up(), reason="llm-serve offline on 127.0.0.1:8100")
def test_live_chat_roundtrip():
    import time
    with ChatClient() as client:
        t0 = time.time()
        resp = client.ask("Say exactly OK.", system="Reply with exactly the word OK.", max_tokens=4, temperature=0)
        elapsed = time.time() - t0
        assert resp.model == DEFAULT_MODEL
        assert "OK" in resp.text.upper()
        assert elapsed > 0
        print(f"[integration] chat roundtrip in {elapsed:.1f}s -> {resp.text!r}")


@pytest.mark.integration
@pytest.mark.skipif(not _llm_up(), reason="llm-serve offline on 127.0.0.1:8100")
def test_live_health_and_models():
    with ChatClient() as client:
        assert client.is_online() is True
        assert DEFAULT_MODEL in client.models()