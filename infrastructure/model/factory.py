"""根据 ModelConfig 创建协议适配器和统一 ModelClient。"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping

from infrastructure.model.client import ModelClient
from infrastructure.model.config import ModelConfig
from infrastructure.model.contracts import ModelConfigurationError, ModelProvider
from infrastructure.model.providers import OpenAICompatibleProvider

ModelProviderBuilder = Callable[[ModelConfig, Mapping[str, str]], ModelProvider]


class ModelClientFactory:
    """协议注册表只负责装配，不在业务调用时反复判断供应商。"""

    def __init__(self) -> None:
        self._builders: dict[str, ModelProviderBuilder] = {
            "openai_compatible": _build_openai_compatible,
        }

    def register(self, protocol: str, builder: ModelProviderBuilder) -> None:
        key = str(protocol).strip()
        if not key or key in self._builders:
            raise ValueError("model protocol must be new and non-empty")
        self._builders[key] = builder

    def create(
        self,
        config: ModelConfig,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> ModelClient:
        if not config.enabled:
            raise ModelConfigurationError("cannot create a disabled model client")
        builder = self._builders.get(config.protocol)
        if builder is None:
            raise ModelConfigurationError(f"unsupported model protocol: {config.protocol}")
        provider = builder(config, os.environ if environ is None else environ)
        return ModelClient(config, provider)


def build_model_client(
    config: ModelConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> ModelClient:
    """使用内置协议注册表创建 Client。"""

    return ModelClientFactory().create(config, environ=environ)


def _build_openai_compatible(config: ModelConfig, environ: Mapping[str, str]) -> ModelProvider:
    api_key = ""
    if config.api_key_env:
        api_key = str(environ.get(config.api_key_env, "")).strip()
        if not api_key:
            raise ModelConfigurationError(f"model credential environment variable is missing: {config.api_key_env}")
    return OpenAICompatibleProvider(
        provider_name=config.provider,
        base_url=config.base_url,
        api_key=api_key,
        timeout_seconds=config.timeout_seconds,
    )


__all__ = ["ModelClientFactory", "ModelProviderBuilder", "build_model_client"]
