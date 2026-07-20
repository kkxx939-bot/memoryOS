"""统一模型 Client：补齐默认配置、执行有限重试并归一化调用结果。"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace

from infrastructure.model.config import ModelConfig
from infrastructure.model.contracts import (
    ChatMessage,
    ChatRequest,
    ModelClientError,
    ModelConfigurationError,
    ModelProvider,
    ModelResponse,
    ModelResponseError,
    ModelTransportError,
)


class ModelClient:
    """对领域暴露稳定调用面，不包含 Prompt 或记忆业务判断。"""

    def __init__(
        self,
        config: ModelConfig,
        provider: ModelProvider,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not config.enabled:
            raise ModelConfigurationError("model client requires an enabled model config")
        self.config = config
        self.provider = provider
        self._sleep = sleep

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def is_remote(self) -> bool:
        return bool(getattr(self.provider, "is_remote", True))

    def complete(self, request: ChatRequest | str) -> ModelResponse:
        """执行一次逻辑调用；只对明确标记为可重试的失败进行有限重试。"""

        normalized = self._request(request)
        attempts = self.config.max_retries + 1
        for attempt in range(attempts):
            try:
                response = self.provider.complete(normalized)
            except ModelClientError as exc:
                if not exc.retryable or attempt + 1 >= attempts:
                    raise
                self._sleep(min(0.25 * (2**attempt), 2.0))
                continue
            except (TimeoutError, ConnectionError, OSError) as exc:
                failure = ModelTransportError(str(exc) or type(exc).__name__)
                if attempt + 1 >= attempts:
                    raise failure from exc
                self._sleep(min(0.25 * (2**attempt), 2.0))
                continue
            if not isinstance(response, ModelResponse) or not response.text.strip():
                raise ModelResponseError("model provider returned an invalid normalized response")
            return response
        raise AssertionError("model retry loop exhausted without a result")  # pragma: no cover

    def health_check(self) -> dict[str, object]:
        """显式探测 Provider，不在 Client 构造或 Runtime 启动时自动访问网络。"""

        result = self.provider.health_check()
        if not isinstance(result, dict):
            raise ModelResponseError("model health check must return an object")
        return dict(result)

    def _request(self, request: ChatRequest | str) -> ChatRequest:
        if isinstance(request, str):
            if not request.strip():
                raise ValueError("model prompt cannot be empty")
            return ChatRequest(
                messages=(ChatMessage(role="user", content=request),),
                model=self.config.model,
            )
        if not isinstance(request, ChatRequest) or not request.messages:
            raise TypeError("model request must be ChatRequest or non-empty text")
        if any(not item.role.strip() or not item.content.strip() for item in request.messages):
            raise ValueError("model messages require non-empty role and content")
        return request if request.model else replace(request, model=self.config.model)


__all__ = ["ModelClient"]
