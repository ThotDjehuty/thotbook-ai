"""Chat client for the local thotbook-ai inference server.

Speaks the OpenAI-compatible surface exposed by ``llm-serve``
(``/health``, ``/v1/models``, ``/v1/chat/completions``). Everything runs
on your own machine — no API key, no billing, fully free by design.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8100/v1"
DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_TIMEOUT = 600.0


@dataclass
class ChatMessage:
    """A single turn in a chat conversation."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    """Parsed ``/v1/chat/completions`` answer."""

    text: str
    model: str
    created: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


class ChatClient:
    """Thread-safe client for the local OpenAI-compatible server.

    Args:
        base_url: base URL of the server, typically ``http://127.0.0.1:8100/v1``.
        model: model id advertised by the server.
        timeout: per-request timeout in seconds (CPU inference is slow —
            keep it generous; default 600s).
        session: optional ``requests.Session`` to reuse a connection pool.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        self.root_url = root.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._session = session if session is not None else requests.Session()

    # -- server introspection ------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Return the ``/health`` payload (``{"status": "ok"}`` when up)."""
        resp = self._session.get(f"{self.root_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def models(self) -> list[str]:
        """List model ids advertised by ``/v1/models``."""
        resp = self._session.get(f"{self.base_url}/models", timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return [m["id"] for m in data.get("data", [])]

    def is_online(self) -> bool:
        """Return True when the server answers ``/health`` with status ok."""
        try:
            return self.health().get("status") == "ok"
        except (requests.RequestException, ValueError):
            return False

    # -- chat ----------------------------------------------------------------

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        """Run a chat completion against the open-source model.

        Args:
            messages: ordered conversation turns.
            max_tokens: generation budget (server default when None).
            temperature: sampling temperature (server default when None).
            top_p: nucleus sampling (server default when None).
            model: override the configured model id.
        """
        if not messages:
            raise ValueError("`messages` must contain at least one message")

        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": [m.to_dict() for m in messages],
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p

        resp = self._session.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return ChatResponse(
            text=text,
            model=data.get("model", self.model),
            created=data.get("created", 0),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            raw=data,
        )

    def ask(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> ChatResponse:
        """One-shot convenience: send a prompt, get the assistant answer."""
        messages: list[ChatMessage] = []
        if system:
            messages.append(ChatMessage(role="system", content=system))
        messages.append(ChatMessage(role="user", content=prompt))
        return self.complete(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> "ChatClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def render_json_payload(payload: dict[str, Any]) -> str:
    """Pretty-print a payload for logs / the ``chat --verbose`` flag."""
    return json.dumps(payload, indent=2)