"""模型配置装配、统一 Client、公共契约和协议适配器。"""

from infrastructure.model.client import ModelClient
from infrastructure.model.config import ModelConfig
from infrastructure.model.contracts import (
    ChatMessage,
    ChatRequest,
    ModelAuthenticationError,
    ModelClientError,
    ModelConfigurationError,
    ModelProvider,
    ModelRateLimitError,
    ModelResponse,
    ModelResponseError,
    ModelTransportError,
)
from infrastructure.model.factory import ModelClientFactory, build_model_client

__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ModelAuthenticationError",
    "ModelClient",
    "ModelConfig",
    "ModelClientError",
    "ModelClientFactory",
    "ModelConfigurationError",
    "ModelProvider",
    "ModelRateLimitError",
    "ModelResponse",
    "ModelResponseError",
    "ModelTransportError",
    "build_model_client",
]
