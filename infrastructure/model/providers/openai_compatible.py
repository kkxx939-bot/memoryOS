"""OpenAI Chat Completions 兼容协议适配器。"""

from __future__ import annotations

import ipaddress
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from infrastructure.model.contracts import (
    ChatRequest,
    ModelAuthenticationError,
    ModelRateLimitError,
    ModelResponse,
    ModelResponseError,
    ModelTransportError,
)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """模型鉴权请求禁止重定向，避免凭证被带到其他地址。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201, ARG002
        return None


class OpenAICompatibleProvider:
    """调用 `/chat/completions` 并转成统一 ModelResponse。"""

    def __init__(
        self,
        *,
        provider_name: str,
        base_url: str,
        api_key: str = "",
        timeout_seconds: float = 30.0,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = float(timeout_seconds)
        self._open = opener or urllib.request.build_opener(_NoRedirectHandler()).open
        self.is_remote = _is_remote_url(self.base_url)

    def complete(self, request: ChatRequest) -> ModelResponse:
        started = time.monotonic()
        payload: dict[str, object] = {
            "model": request.model,
            "messages": [{"role": item.role, "content": item.content} for item in request.messages],
            "temperature": request.temperature,
        }
        response = self._request("POST", "/chat/completions", payload=payload)
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ModelResponseError("openai-compatible response has no choices")
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ModelResponseError("openai-compatible response has no message content")
        usage = response.get("usage")
        return ModelResponse(
            text=text,
            model=str(response.get("model") or request.model or ""),
            provider=self.provider_name,
            prompt_version=request.prompt_version,
            usage=dict(usage) if isinstance(usage, dict) else {},
            latency_ms=max(0, round((time.monotonic() - started) * 1000)),
            raw=response,
        )

    def health_check(self) -> dict[str, object]:
        self._request("GET", "/models")
        return {"ok": True, "provider": self.provider_name}

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with self._open(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            self._raise_http_error(exc)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            raise ModelTransportError(str(exc) or type(exc).__name__) from exc
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelResponseError("model provider returned malformed JSON") from exc
        if not isinstance(decoded, dict):
            raise ModelResponseError("model provider response must be an object")
        return decoded

    @staticmethod
    def _raise_http_error(exc: urllib.error.HTTPError) -> None:
        status = int(exc.code)
        if status in {401, 403}:
            raise ModelAuthenticationError(f"model provider rejected credentials with HTTP {status}") from exc
        if status == 429:
            raise ModelRateLimitError("model provider rate limit exceeded") from exc
        if status in {408, 425, 500, 502, 503, 504}:
            raise ModelTransportError(f"model provider is temporarily unavailable: HTTP {status}") from exc
        raise ModelResponseError(f"model provider request failed with HTTP {status}") from exc


__all__ = ["OpenAICompatibleProvider"]


def _is_remote_url(url: str) -> bool:
    hostname = str(urlsplit(url).hostname or "").casefold()
    if hostname == "localhost":
        return False
    try:
        return not ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return True
