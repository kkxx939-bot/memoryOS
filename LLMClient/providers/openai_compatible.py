"""支持 SSE 流式输出的 OpenAI Chat Completions 兼容 HTTP 供应商。"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import time
import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from typing import Any, cast
from urllib.parse import urlsplit

from LLMClient.contracts import (
    ChatRequest,
    ModelResponse,
    ModelResponseError,
    ModelStreamEvent,
    ProviderCapabilities,
)
from LLMClient.providers.common import (
    build_openai_payload,
    normalize_response,
    normalize_stream_chunk,
)
from LLMClient.retry import normalize_provider_error


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """重定向时绝不转发模型鉴权请求头。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201, ARG002
        return None


class _ProviderHTTPError(RuntimeError):
    def __init__(self, status_code: int, message: str, retry_after: float | None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class OpenAICompatibleProvider:
    """规范化一条 OpenAI 兼容路由，不引入供应商专用业务逻辑。"""

    def __init__(
        self,
        *,
        provider_name: str,
        model: str,
        base_url: str,
        api_key: str = "",
        timeout_seconds: float = 30.0,
        max_output_tokens: int | None = None,
        max_response_bytes: int = 8 * 1024 * 1024,
        native_structured_output: bool = False,
        reasoning: bool = False,
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, object] | None = None,
        max_retries: int = 2,
        retry_base_delay_seconds: float = 0.5,
        retry_max_delay_seconds: float = 30.0,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_tokens = max_output_tokens
        self.max_response_bytes = max_response_bytes
        self.configured_reasoning = reasoning
        self.extra_headers = dict(extra_headers or {})
        self.extra_body = dict(extra_body or {})
        self.max_retries = max_retries
        self.retry_base_delay_seconds = retry_base_delay_seconds
        self.retry_max_delay_seconds = retry_max_delay_seconds
        self._open = opener or urllib.request.build_opener(_NoRedirectHandler()).open
        self.is_remote = _is_remote_url(self.base_url)
        self.capabilities = ProviderCapabilities(
            async_completion=True,
            streaming=True,
            tools=True,
            native_structured_output=native_structured_output,
            reasoning=reasoning,
        )

    def complete(self, request: ChatRequest) -> ModelResponse:
        started = time.monotonic()
        payload = self._payload(request, stream=False)
        response = self._request_json("POST", "/chat/completions", payload=payload)
        return normalize_response(
            response,
            provider=self.provider_name,
            configured_model=self.model,
            prompt_version=request.prompt_version,
            started=started,
        )

    async def complete_async(self, request: ChatRequest) -> ModelResponse:
        return await asyncio.to_thread(self.complete, request)

    def stream(self, request: ChatRequest) -> Iterator[ModelStreamEvent]:
        payload = self._payload(request, stream=True)
        yield from self._request_stream("POST", "/chat/completions", payload=payload)

    async def stream_async(self, request: ChatRequest) -> AsyncIterator[ModelStreamEvent]:
        iterator = iter(self.stream(request))
        while True:
            item = await asyncio.to_thread(_next_or_end, iterator)
            if item is _STREAM_END:
                break
            yield cast(ModelStreamEvent, item)

    def health_check(self) -> Mapping[str, object]:
        self._request_json("GET", "/models")
        return {"ok": True, "provider": self.provider_name, "model": self.model}

    def _payload(self, request: ChatRequest, *, stream: bool) -> dict[str, object]:
        return build_openai_payload(
            request,
            model=self.model,
            capabilities=self.capabilities,
            configured_reasoning=self.configured_reasoning,
            default_max_output_tokens=self.max_output_tokens,
            extra_body=self.extra_body,
            stream=stream,
        )

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        request = self._build_request(method, path, payload=payload)
        try:
            with self._open(request, timeout=self.timeout_seconds) as response:
                raw = _read_limited(response, self.max_response_bytes)
        except urllib.error.HTTPError as exc:
            self._raise_http_error(exc)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            raise normalize_provider_error(exc) from exc
        try:
            decoded = json.loads(raw.decode("utf-8"), parse_constant=_reject_non_finite)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ModelResponseError("model provider returned malformed JSON") from exc
        if not isinstance(decoded, dict):
            raise ModelResponseError("model provider response must be an object")
        return decoded

    def _request_stream(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, object],
    ) -> Iterator[ModelStreamEvent]:
        request = self._build_request(method, path, payload=payload)
        try:
            with self._open(request, timeout=self.timeout_seconds) as response:
                yield from self._iter_sse(response)
        except urllib.error.HTTPError as exc:
            self._raise_http_error(exc)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            raise normalize_provider_error(exc) from exc

    def _iter_sse(self, response: Any) -> Iterator[ModelStreamEvent]:
        total_bytes = 0
        data_lines: list[str] = []
        finish_reason: str | None = None
        for raw_line in response:
            if not isinstance(raw_line, bytes):
                raise ModelResponseError("model stream returned a non-byte frame")
            total_bytes += len(raw_line)
            if total_bytes > self.max_response_bytes:
                raise ModelResponseError("model stream exceeded configured response size")
            try:
                line = raw_line.decode("utf-8").rstrip("\r\n")
            except UnicodeDecodeError as exc:
                raise ModelResponseError("model stream returned non-UTF-8 data") from exc
            if not line:
                for event in self._decode_sse_event(data_lines):
                    if event.kind == "done":
                        finish_reason = finish_reason or event.finish_reason
                        continue
                    yield event
                data_lines.clear()
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            elif line.startswith("{"):
                data_lines.append(line)
        for event in self._decode_sse_event(data_lines):
            if event.kind == "done":
                finish_reason = finish_reason or event.finish_reason
                continue
            yield event
        yield ModelStreamEvent(kind="done", finish_reason=finish_reason or "stop")

    @staticmethod
    def _decode_sse_event(data_lines: list[str]) -> tuple[ModelStreamEvent, ...]:
        if not data_lines:
            return ()
        data = "\n".join(data_lines).strip()
        if data == "[DONE]":
            return (ModelStreamEvent(kind="done", finish_reason="stop"),)
        try:
            decoded = json.loads(data, parse_constant=_reject_non_finite)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ModelResponseError("model stream returned malformed JSON") from exc
        if not isinstance(decoded, dict):
            raise ModelResponseError("model stream event must be an object")
        return normalize_stream_chunk(decoded)

    def _build_request(
        self,
        method: str,
        path: str,
        *,
        payload: Mapping[str, object] | None = None,
    ) -> urllib.request.Request:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Accept": "application/json", **self.extra_headers}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return urllib.request.Request(
            self.base_url + path,
            data=body,
            headers=headers,
            method=method,
        )

    def _raise_http_error(self, exc: urllib.error.HTTPError) -> None:
        raw = exc.read(64 * 1024)
        detail = raw.decode("utf-8", errors="replace").strip()
        message = f"model provider request failed with HTTP {int(exc.code)}"
        if detail:
            message = f"{message}: {detail[:2048]}"
        retry_after = _header_float(exc.headers, "Retry-After")
        failure = _ProviderHTTPError(int(exc.code), message, retry_after)
        raise normalize_provider_error(failure) from exc


class _StreamEnd:
    pass


_STREAM_END = _StreamEnd()


def _next_or_end(iterator: Iterator[ModelStreamEvent]) -> ModelStreamEvent | _StreamEnd:
    return next(iterator, _STREAM_END)


def _read_limited(response: Any, limit: int) -> bytes:
    raw = response.read(limit + 1)
    if len(raw) > limit:
        raise ModelResponseError("model provider response exceeded configured size")
    return raw


def _header_float(headers: object, name: str) -> float | None:
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    try:
        value = getter(name)
    except TypeError:
        return None
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def _reject_non_finite(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _is_remote_url(url: str) -> bool:
    hostname = str(urlsplit(url).hostname or "").casefold()
    if hostname == "localhost":
        return False
    try:
        return not ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return True


__all__ = ["OpenAICompatibleProvider"]
