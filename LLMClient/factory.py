"""LLMClient 的供应商注册表和有序路由构建逻辑。"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping

from LLMClient.client import LLMClient
from LLMClient.config import ModelConfig
from LLMClient.contracts import ModelConfigurationError, ModelProvider
from LLMClient.providers import LiteLLMProvider, OpenAICompatibleProvider

ModelProviderBuilder = Callable[[ModelConfig, Mapping[str, str]], ModelProvider]


class LLMClientFactory:
    """统一解析协议适配器，避免业务代码依赖供应商名称。"""

    def __init__(self) -> None:
        self._builders: dict[str, ModelProviderBuilder] = {
            "openai_compatible": _build_openai_compatible,
            "litellm": _build_litellm,
        }

    def register(self, protocol: str, builder: ModelProviderBuilder) -> None:
        key = str(protocol).strip().lower()
        if not key or key in self._builders:
            raise ValueError("model protocol must be new and non-empty")
        if not callable(builder):
            raise TypeError("model provider builder must be callable")
        self._builders[key] = builder

    def create(
        self,
        config: ModelConfig,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> LLMClient:
        if not config.enabled:
            raise ModelConfigurationError("cannot create a disabled model client")
        environment = os.environ if environ is None else environ
        providers = tuple(self._build(route, environment) for route in _flatten_routes(config))
        return LLMClient(config, providers)

    def _build(self, config: ModelConfig, environ: Mapping[str, str]) -> ModelProvider:
        builder = self._builders.get(config.protocol)
        if builder is None:
            raise ModelConfigurationError(f"unsupported model protocol: {config.protocol}")
        return builder(config, environ)


def build_llm_client(
    config: ModelConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> LLMClient:
    return LLMClientFactory().create(config, environ=environ)


def _flatten_routes(config: ModelConfig) -> tuple[ModelConfig, ...]:
    routes: list[ModelConfig] = [config]
    for fallback in config.fallbacks:
        routes.extend(_flatten_routes(fallback))
    return tuple(routes)


def _credential(config: ModelConfig, environ: Mapping[str, str]) -> str:
    if not config.api_key_env:
        return ""
    api_key = str(environ.get(config.api_key_env, "")).strip()
    if not api_key:
        raise ModelConfigurationError(f"model credential environment variable is missing: {config.api_key_env}")
    return api_key


def _build_openai_compatible(
    config: ModelConfig,
    environ: Mapping[str, str],
) -> ModelProvider:
    return OpenAICompatibleProvider(
        provider_name=config.provider,
        model=config.model,
        base_url=config.base_url,
        api_key=_credential(config, environ),
        timeout_seconds=config.timeout_seconds,
        max_output_tokens=config.max_output_tokens,
        max_response_bytes=config.max_response_bytes,
        native_structured_output=config.native_structured_output,
        reasoning=config.reasoning,
        extra_headers=config.extra_headers,
        extra_body=config.extra_body,
        max_retries=config.max_retries,
        retry_base_delay_seconds=config.retry_base_delay_seconds,
        retry_max_delay_seconds=config.retry_max_delay_seconds,
    )


def _build_litellm(config: ModelConfig, environ: Mapping[str, str]) -> ModelProvider:
    return LiteLLMProvider(
        provider_name=config.provider,
        model=config.model,
        api_key=_credential(config, environ),
        base_url=config.base_url,
        timeout_seconds=config.timeout_seconds,
        max_output_tokens=config.max_output_tokens,
        max_response_bytes=config.max_response_bytes,
        native_structured_output=config.native_structured_output,
        reasoning=config.reasoning,
        extra_headers=config.extra_headers,
        extra_body=config.extra_body,
        max_retries=config.max_retries,
        retry_base_delay_seconds=config.retry_base_delay_seconds,
        retry_max_delay_seconds=config.retry_max_delay_seconds,
    )


__all__ = ["LLMClientFactory", "ModelProviderBuilder", "build_llm_client"]
