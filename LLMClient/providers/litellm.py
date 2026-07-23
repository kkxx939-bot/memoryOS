"""可选的 LiteLLM 适配器，用于扩展 OpenAI 兼容端点之外的供应商支持。"""

from __future__ import annotations

import importlib
import json
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from types import ModuleType

from LLMClient.contracts import (
    ChatRequest,
    ModelDependencyError,
    ModelResponse,
    ModelResponseError,
    ModelStreamEvent,
    ProviderCapabilities,
)
from LLMClient.providers.common import (
    build_openai_payload,
    normalize_response,
    normalize_stream_chunk,
    object_to_mapping,
)
from LLMClient.retry import normalize_provider_error


class LiteLLMProvider:
    """将 LiteLLM 作为可选协议适配器使用，不把它设为基础依赖。"""

    def __init__(
        self,
        *,
        provider_name: str,
        model: str,
        api_key: str = "",
        base_url: str = "",
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
    ) -> None:
        self.provider_name = provider_name
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.max_response_bytes = max_response_bytes
        self.configured_reasoning = reasoning
        self.extra_headers = dict(extra_headers or {})
        self.extra_body = dict(extra_body or {})
        self.max_retries = max_retries
        self.retry_base_delay_seconds = retry_base_delay_seconds
        self.retry_max_delay_seconds = retry_max_delay_seconds
        self.is_remote = True
        self.capabilities = ProviderCapabilities(
            async_completion=True,
            streaming=True,
            tools=True,
            native_structured_output=native_structured_output,
            reasoning=reasoning,
        )

    def complete(self, request: ChatRequest) -> ModelResponse:
        started = time.monotonic()
        module = self._module()
        try:
            response = module.completion(**self._kwargs(request, stream=False))
            normalized = object_to_mapping(response)
            self._require_bounded(normalized)
            return normalize_response(
                normalized,
                provider=self.provider_name,
                configured_model=self.model,
                prompt_version=request.prompt_version,
                started=started,
            )
        except Exception as exc:
            if isinstance(exc, (ModelDependencyError, ModelResponseError)):
                raise
            raise normalize_provider_error(exc) from exc

    async def complete_async(self, request: ChatRequest) -> ModelResponse:
        started = time.monotonic()
        module = self._module()
        try:
            response = await module.acompletion(**self._kwargs(request, stream=False))
            normalized = object_to_mapping(response)
            self._require_bounded(normalized)
            return normalize_response(
                normalized,
                provider=self.provider_name,
                configured_model=self.model,
                prompt_version=request.prompt_version,
                started=started,
            )
        except Exception as exc:
            if isinstance(exc, (ModelDependencyError, ModelResponseError)):
                raise
            raise normalize_provider_error(exc) from exc

    def stream(self, request: ChatRequest) -> Iterator[ModelStreamEvent]:
        module = self._module()
        finish_reason: str | None = None
        try:
            response = module.completion(**self._kwargs(request, stream=True))
            for chunk in response:
                normalized = object_to_mapping(chunk)
                self._require_bounded(normalized)
                for event in normalize_stream_chunk(normalized):
                    if event.kind == "done":
                        finish_reason = finish_reason or event.finish_reason
                        continue
                    yield event
        except Exception as exc:
            if isinstance(exc, (ModelDependencyError, ModelResponseError)):
                raise
            raise normalize_provider_error(exc) from exc
        yield ModelStreamEvent(kind="done", finish_reason=finish_reason or "stop")

    async def stream_async(self, request: ChatRequest) -> AsyncIterator[ModelStreamEvent]:
        module = self._module()
        finish_reason: str | None = None
        try:
            response = await module.acompletion(**self._kwargs(request, stream=True))
            async for chunk in response:
                normalized = object_to_mapping(chunk)
                self._require_bounded(normalized)
                for event in normalize_stream_chunk(normalized):
                    if event.kind == "done":
                        finish_reason = finish_reason or event.finish_reason
                        continue
                    yield event
        except Exception as exc:
            if isinstance(exc, (ModelDependencyError, ModelResponseError)):
                raise
            raise normalize_provider_error(exc) from exc
        yield ModelStreamEvent(kind="done", finish_reason=finish_reason or "stop")

    def health_check(self) -> Mapping[str, object]:
        module = self._module()
        resolver = getattr(module, "get_llm_provider", None)
        if callable(resolver):
            resolver(model=self.model)
        return {
            "ok": True,
            "provider": self.provider_name,
            "model": self.model,
            "network_checked": False,
        }

    def _kwargs(self, request: ChatRequest, *, stream: bool) -> dict[str, object]:
        kwargs = build_openai_payload(
            request,
            model=self.model,
            capabilities=self.capabilities,
            configured_reasoning=self.configured_reasoning,
            default_max_output_tokens=self.max_output_tokens,
            extra_body=self.extra_body,
            stream=stream,
        )
        kwargs["timeout"] = self.timeout_seconds
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url
        if self.extra_headers:
            kwargs["extra_headers"] = self.extra_headers
        return kwargs

    def _require_bounded(self, source: Mapping[str, object]) -> None:
        size = len(json.dumps(source, ensure_ascii=False, default=str).encode("utf-8"))
        if size > self.max_response_bytes:
            raise ModelResponseError("model provider response exceeded configured size")

    @staticmethod
    def _module() -> ModuleType:
        try:
            return importlib.import_module("litellm")
        except ImportError as exc:
            raise ModelDependencyError("LiteLLM protocol requires the optional 'litellm' package") from exc


__all__ = ["LiteLLMProvider"]
