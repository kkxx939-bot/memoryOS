"""通用模型调用的数据契约、Provider 协议和有限失败类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ChatMessage:
    """一条与具体供应商无关的对话消息。"""

    role: str
    content: str


@dataclass(frozen=True)
class ChatRequest:
    """统一模型请求；领域 metadata 不会自动发送给外部供应商。"""

    messages: tuple[ChatMessage, ...]
    model: str | None = None
    temperature: float = 0.0
    prompt_version: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    """供应商响应归一化后的最小结果。"""

    text: str
    model: str
    provider: str
    prompt_version: str | None = None
    usage: dict[str, object] = field(default_factory=dict)
    latency_ms: int | None = None
    raw: dict[str, object] | None = None


class ModelProvider(Protocol):
    """具体协议适配器必须实现的边界。"""

    provider_name: str
    is_remote: bool

    def complete(self, request: ChatRequest) -> ModelResponse: ...

    def health_check(self) -> dict[str, object]: ...


class ModelClientError(RuntimeError):
    """统一模型调用失败，供上层判断是否可以安全重试。"""

    retryable = False
    code = "MODEL_CLIENT_ERROR"


class ModelConfigurationError(ModelClientError):
    code = "MODEL_CONFIGURATION"


class ModelAuthenticationError(ModelClientError):
    code = "MODEL_AUTHENTICATION"


class ModelTransportError(ModelClientError):
    retryable = True
    code = "MODEL_TRANSPORT"


class ModelRateLimitError(ModelTransportError):
    code = "MODEL_RATE_LIMIT"


class ModelResponseError(ModelClientError):
    code = "MODEL_RESPONSE"


__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ModelAuthenticationError",
    "ModelClientError",
    "ModelConfigurationError",
    "ModelProvider",
    "ModelRateLimitError",
    "ModelResponse",
    "ModelResponseError",
    "ModelTransportError",
]
