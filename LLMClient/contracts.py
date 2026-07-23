"""文本、工具调用、结构化输出和流式输出的供应商无关契约。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

MessageRole = Literal["system", "developer", "user", "assistant", "tool"]
StreamEventKind = Literal[
    "content_delta",
    "reasoning_delta",
    "tool_call_delta",
    "usage",
    "done",
]


@dataclass(frozen=True)
class ToolCall:
    """模型返回的一次规范化函数调用。"""

    id: str
    name: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("tool call id must be non-empty")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("tool call name must be non-empty")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("tool call arguments must be an object")
        object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(frozen=True)
class ToolDefinition:
    """与供应商无关的函数工具声明。"""

    name: str
    description: str
    parameters: Mapping[str, object]
    strict: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("tool name must be non-empty")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("tool description must be non-empty")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("tool parameters must be a JSON Schema object")
        if not isinstance(self.strict, bool):
            raise TypeError("tool strict must be boolean")
        object.__setattr__(self, "parameters", dict(self.parameters))


@dataclass(frozen=True)
class ChatMessage:
    """保留助手工具调用和工具结果的规范化消息。"""

    role: MessageRole
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported chat message role: {self.role}")
        if self.content is not None and not isinstance(self.content, str):
            raise TypeError("message content must be text or null")
        if self.name is not None and (not isinstance(self.name, str) or not self.name.strip()):
            raise ValueError("message name must be non-empty when provided")
        if self.tool_call_id is not None and (not isinstance(self.tool_call_id, str) or not self.tool_call_id.strip()):
            raise ValueError("tool_call_id must be non-empty when provided")
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if self.role == "tool":
            if not self.tool_call_id:
                raise ValueError("tool result messages require tool_call_id")
            if self.content is None:
                raise ValueError("tool result messages require content")
            if self.tool_calls:
                raise ValueError("tool result messages cannot contain tool calls")
        elif self.tool_call_id is not None:
            raise ValueError("tool_call_id is only valid for tool result messages")
        if self.tool_calls and self.role != "assistant":
            raise ValueError("only assistant messages may contain tool calls")
        if self.content is None and not self.tool_calls:
            raise ValueError("messages require content or assistant tool calls")


@dataclass(frozen=True)
class ResponseFormat:
    """向原生支持结构化输出的供应商请求使用的 JSON Schema。"""

    name: str
    schema: Mapping[str, object]
    strict: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("response format name must be non-empty")
        if not isinstance(self.schema, Mapping) or not self.schema:
            raise ValueError("response format schema must be a non-empty object")
        if not isinstance(self.strict, bool):
            raise TypeError("response format strict must be boolean")
        object.__setattr__(self, "schema", dict(self.schema))


@dataclass(frozen=True)
class ReasoningOptions:
    """可移植的推理意图；供应商专用开关应放入路由的 extra_body。"""

    effort: Literal["minimal", "low", "medium", "high"] | None = None

    def __post_init__(self) -> None:
        if self.effort not in {None, "minimal", "low", "medium", "high"}:
            raise ValueError("unsupported reasoning effort")


@dataclass(frozen=True)
class ChatRequest:
    """不依赖具体供应商或模型路由的一次逻辑模型请求。"""

    messages: tuple[ChatMessage, ...]
    temperature: float | None = 0.0
    max_output_tokens: int | None = None
    tools: tuple[ToolDefinition, ...] = ()
    tool_choice: str | Mapping[str, object] | None = None
    response_format: ResponseFormat | None = None
    reasoning: ReasoningOptions | None = None
    prompt_version: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.messages or any(not isinstance(item, ChatMessage) for item in self.messages):
            raise ValueError("chat request requires normalized messages")
        if self.temperature is not None:
            if isinstance(self.temperature, bool) or not isinstance(self.temperature, int | float):
                raise TypeError("temperature must be numeric or null")
            if not 0 <= float(self.temperature) <= 2:
                raise ValueError("temperature must be between zero and two")
        if self.max_output_tokens is not None:
            if (
                isinstance(self.max_output_tokens, bool)
                or not isinstance(self.max_output_tokens, int)
                or self.max_output_tokens <= 0
            ):
                raise ValueError("max_output_tokens must be a positive integer")
        if any(not isinstance(tool, ToolDefinition) for tool in self.tools):
            raise TypeError("tools must contain ToolDefinition values")
        if self.tool_choice is not None and not self.tools:
            raise ValueError("tool_choice requires at least one tool")
        if self.response_format is not None and not isinstance(self.response_format, ResponseFormat):
            raise TypeError("response_format must be a ResponseFormat")
        if self.reasoning is not None and not isinstance(self.reasoning, ReasoningOptions):
            raise TypeError("reasoning must be ReasoningOptions")


@dataclass(frozen=True)
class TokenUsage:
    """规范化的 Token 用量统计，可附带供应商明细。"""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "details", dict(self.details))


@dataclass(frozen=True)
class ModelResponse:
    """一次规范化最终响应，包含工具调用和终止元数据。"""

    content: str | None
    model: str
    provider: str
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = "stop"
    reasoning_content: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    prompt_version: str | None = None
    latency_ms: int | None = None
    raw: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if self.content is not None and not isinstance(self.content, str):
            raise TypeError("model response content must be text or null")
        if not self.content and not self.tool_calls:
            raise ValueError("model response requires content or tool calls")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model response model must be non-empty")
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("model response provider must be non-empty")
        if not isinstance(self.finish_reason, str) or not self.finish_reason.strip():
            raise ValueError("model response finish_reason must be non-empty")
        if self.reasoning_content is not None and not isinstance(self.reasoning_content, str):
            raise TypeError("reasoning_content must be text or null")
        if not isinstance(self.usage, TokenUsage):
            raise TypeError("usage must be TokenUsage")
        if self.raw is not None:
            object.__setattr__(self, "raw", dict(self.raw))


@dataclass(frozen=True)
class ModelStreamEvent:
    """一次规范化流事件；工具参数在结束事件前始终保持增量形式。"""

    kind: StreamEventKind
    content_delta: str | None = None
    reasoning_delta: str | None = None
    tool_call_index: int | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_arguments_delta: str | None = None
    usage: TokenUsage | None = None
    finish_reason: str | None = None
    raw: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        allowed = {"content_delta", "reasoning_delta", "tool_call_delta", "usage", "done"}
        if self.kind not in allowed:
            raise ValueError(f"unsupported stream event kind: {self.kind}")
        if self.raw is not None:
            object.__setattr__(self, "raw", dict(self.raw))


@dataclass(frozen=True)
class ProviderCapabilities:
    """用于适配请求的能力声明，避免按供应商名称编写条件分支。"""

    async_completion: bool = True
    streaming: bool = True
    tools: bool = True
    native_structured_output: bool = False
    reasoning: bool = False


class ModelProvider(Protocol):
    """具体供应商在此实现统一的模型边界。"""

    provider_name: str
    model: str
    is_remote: bool
    capabilities: ProviderCapabilities
    max_retries: int
    retry_base_delay_seconds: float
    retry_max_delay_seconds: float

    def complete(self, request: ChatRequest) -> ModelResponse: ...

    async def complete_async(self, request: ChatRequest) -> ModelResponse: ...

    def stream(self, request: ChatRequest) -> Iterator[ModelStreamEvent]: ...

    def stream_async(self, request: ChatRequest) -> AsyncIterator[ModelStreamEvent]: ...

    def health_check(self) -> Mapping[str, object]: ...


class ModelClientError(RuntimeError):
    """携带重试和故障转移策略元数据的基础异常。"""

    retryable = False
    failover_eligible = False
    code = "MODEL_CLIENT_ERROR"

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class ModelConfigurationError(ModelClientError):
    code = "MODEL_CONFIGURATION"


class ModelDependencyError(ModelConfigurationError):
    failover_eligible = True
    code = "MODEL_DEPENDENCY"


class ModelAuthenticationError(ModelClientError):
    failover_eligible = True
    code = "MODEL_AUTHENTICATION"


class ModelPermissionError(ModelClientError):
    failover_eligible = True
    code = "MODEL_PERMISSION"


class ModelTransportError(ModelClientError):
    retryable = True
    failover_eligible = True
    code = "MODEL_TRANSPORT"


class ModelRateLimitError(ModelTransportError):
    code = "MODEL_RATE_LIMIT"


class ModelQuotaError(ModelClientError):
    failover_eligible = True
    code = "MODEL_QUOTA"


class ModelInputTooLargeError(ModelClientError):
    code = "MODEL_INPUT_TOO_LARGE"


class ModelContentSafetyError(ModelClientError):
    code = "MODEL_CONTENT_SAFETY"


class ModelResponseError(ModelClientError):
    failover_eligible = True
    code = "MODEL_RESPONSE"


class ModelStructuredOutputError(ModelClientError):
    code = "MODEL_STRUCTURED_OUTPUT"


__all__ = [
    "ChatMessage",
    "ChatRequest",
    "MessageRole",
    "ModelAuthenticationError",
    "ModelClientError",
    "ModelConfigurationError",
    "ModelContentSafetyError",
    "ModelDependencyError",
    "ModelInputTooLargeError",
    "ModelPermissionError",
    "ModelProvider",
    "ModelQuotaError",
    "ModelRateLimitError",
    "ModelResponse",
    "ModelResponseError",
    "ModelStreamEvent",
    "ModelStructuredOutputError",
    "ModelTransportError",
    "ProviderCapabilities",
    "ReasoningOptions",
    "ResponseFormat",
    "StreamEventKind",
    "TokenUsage",
    "ToolCall",
    "ToolDefinition",
]
