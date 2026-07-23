"""支持并发、重试、故障转移、异步调用和流式输出的统一模型客户端。"""

from __future__ import annotations

import asyncio
import threading
import time
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from dataclasses import replace

from LLMClient.config import ModelConfig
from LLMClient.contracts import (
    ChatMessage,
    ChatRequest,
    ModelClientError,
    ModelConfigurationError,
    ModelProvider,
    ModelResponse,
    ModelResponseError,
    ModelStreamEvent,
)
from LLMClient.retry import normalize_provider_error, retry_delay


class LLMClient:
    """在有序供应商路由集合上提供稳定的调用接口。"""

    def __init__(
        self,
        config: ModelConfig,
        providers: Sequence[ModelProvider],
        *,
        sleep: Callable[[float], None] = time.sleep,
        async_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not config.enabled:
            raise ModelConfigurationError("LLMClient requires an enabled model config")
        normalized_providers = tuple(providers)
        if not normalized_providers:
            raise ModelConfigurationError("LLMClient requires at least one provider route")
        self.config = config
        self.providers = normalized_providers
        self._sleep = sleep
        self._async_sleep = async_sleep
        self._sync_slots = threading.BoundedSemaphore(config.max_concurrent)
        self._async_slots: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )
        self._async_slots_lock = threading.Lock()

    @property
    def provider(self) -> ModelProvider:
        return self.providers[0]

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name

    @property
    def model(self) -> str:
        return self.provider.model

    @property
    def is_remote(self) -> bool:
        return self.provider.is_remote

    def complete(self, request: ChatRequest | str) -> ModelResponse:
        normalized = self._request(request)
        with self._sync_slots:
            return self._complete_sync(normalized)

    async def complete_async(self, request: ChatRequest | str) -> ModelResponse:
        normalized = self._request(request)
        async with self._async_slot():
            return await self._complete_async(normalized)

    def stream(self, request: ChatRequest | str) -> Iterator[ModelStreamEvent]:
        normalized = self._request(request)
        return self._stream_sync(normalized)

    async def stream_async(self, request: ChatRequest | str) -> AsyncIterator[ModelStreamEvent]:
        normalized = self._request(request)
        async with self._async_slot():
            async for event in self._stream_async(normalized):
                yield event

    def health_check(self) -> dict[str, object]:
        routes: list[dict[str, object]] = []
        for provider in self.providers:
            try:
                result = provider.health_check()
                if not isinstance(result, dict):
                    result = dict(result)
                routes.append(dict(result))
            except Exception as exc:
                failure = normalize_provider_error(exc)
                routes.append(
                    {
                        "ok": False,
                        "provider": provider.provider_name,
                        "model": provider.model,
                        "error_code": failure.code,
                    }
                )
        return {"ok": any(bool(route.get("ok")) for route in routes), "routes": routes}

    def _complete_sync(self, request: ChatRequest) -> ModelResponse:
        last_failure: ModelClientError | None = None
        for route_index, provider in enumerate(self.providers):
            for attempt in range(provider.max_retries + 1):
                try:
                    response = provider.complete(request)
                    return self._require_response(response)
                except Exception as exc:
                    failure = normalize_provider_error(exc)
                    last_failure = failure
                    if failure.retryable and attempt < provider.max_retries:
                        self._sleep(self._retry_delay(provider, attempt, failure))
                        continue
                    if self._can_fail_over(failure, route_index):
                        break
                    raise failure from exc
        if last_failure is not None:
            raise last_failure
        raise AssertionError("provider route loop exhausted without a result")  # pragma: no cover

    async def _complete_async(self, request: ChatRequest) -> ModelResponse:
        last_failure: ModelClientError | None = None
        for route_index, provider in enumerate(self.providers):
            for attempt in range(provider.max_retries + 1):
                try:
                    response = await provider.complete_async(request)
                    return self._require_response(response)
                except Exception as exc:
                    failure = normalize_provider_error(exc)
                    last_failure = failure
                    if failure.retryable and attempt < provider.max_retries:
                        await self._async_sleep(self._retry_delay(provider, attempt, failure))
                        continue
                    if self._can_fail_over(failure, route_index):
                        break
                    raise failure from exc
        if last_failure is not None:
            raise last_failure
        raise AssertionError("provider route loop exhausted without a result")  # pragma: no cover

    def _stream_sync(self, request: ChatRequest) -> Iterator[ModelStreamEvent]:
        with self._sync_slots:
            last_failure: ModelClientError | None = None
            for route_index, provider in enumerate(self.providers):
                for attempt in range(provider.max_retries + 1):
                    emitted = False
                    try:
                        for event in provider.stream(request):
                            if not isinstance(event, ModelStreamEvent):
                                raise ModelResponseError("provider returned an invalid stream event")
                            emitted = True
                            yield event
                        if not emitted:
                            raise ModelResponseError("provider returned an empty stream")
                        return
                    except Exception as exc:
                        failure = normalize_provider_error(exc)
                        last_failure = failure
                        if emitted:
                            raise failure from exc
                        if failure.retryable and attempt < provider.max_retries:
                            self._sleep(self._retry_delay(provider, attempt, failure))
                            continue
                        if self._can_fail_over(failure, route_index):
                            break
                        raise failure from exc
            if last_failure is not None:
                raise last_failure

    async def _stream_async(self, request: ChatRequest) -> AsyncIterator[ModelStreamEvent]:
        last_failure: ModelClientError | None = None
        for route_index, provider in enumerate(self.providers):
            for attempt in range(provider.max_retries + 1):
                emitted = False
                try:
                    async for event in provider.stream_async(request):
                        if not isinstance(event, ModelStreamEvent):
                            raise ModelResponseError("provider returned an invalid stream event")
                        emitted = True
                        yield event
                    if not emitted:
                        raise ModelResponseError("provider returned an empty stream")
                    return
                except Exception as exc:
                    failure = normalize_provider_error(exc)
                    last_failure = failure
                    if emitted:
                        raise failure from exc
                    if failure.retryable and attempt < provider.max_retries:
                        await self._async_sleep(self._retry_delay(provider, attempt, failure))
                        continue
                    if self._can_fail_over(failure, route_index):
                        break
                    raise failure from exc
        if last_failure is not None:
            raise last_failure

    def _request(self, request: ChatRequest | str) -> ChatRequest:
        if isinstance(request, str):
            if not request.strip():
                raise ValueError("model prompt cannot be empty")
            request = ChatRequest(messages=(ChatMessage(role="user", content=request),))
        if not isinstance(request, ChatRequest):
            raise TypeError("model request must be ChatRequest or non-empty text")
        if request.max_output_tokens is None and self.config.max_output_tokens is not None:
            return replace(request, max_output_tokens=self.config.max_output_tokens)
        return request

    @staticmethod
    def _require_response(response: object) -> ModelResponse:
        if not isinstance(response, ModelResponse):
            raise ModelResponseError("provider returned an invalid normalized response")
        return response

    @staticmethod
    def _retry_delay(
        provider: ModelProvider,
        attempt: int,
        failure: ModelClientError,
    ) -> float:
        return retry_delay(
            attempt,
            base_delay=provider.retry_base_delay_seconds,
            max_delay=provider.retry_max_delay_seconds,
            error=failure,
        )

    def _can_fail_over(self, failure: ModelClientError, route_index: int) -> bool:
        return failure.failover_eligible and route_index + 1 < len(self.providers)

    def _async_slot(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        with self._async_slots_lock:
            semaphore = self._async_slots.get(loop)
            if semaphore is None:
                semaphore = asyncio.Semaphore(self.config.max_concurrent)
                self._async_slots[loop] = semaphore
            return semaphore


__all__ = ["LLMClient"]
