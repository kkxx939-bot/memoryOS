"""供应商无关的稠密文本向量契约与 LiteLLM 适配器。"""

from __future__ import annotations

import importlib
import math
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import ModuleType
from typing import Protocol, cast

from LLMClient.contracts import (
    ModelConfigurationError,
    ModelDependencyError,
    ModelResponseError,
)
from LLMClient.retry import normalize_provider_error


@dataclass(frozen=True)
class EmbeddingVector:
    """经过有限值和非零范数校验的稠密向量。"""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if isinstance(self.values, list):
            values = tuple(self.values)
        elif isinstance(self.values, tuple):
            values = self.values
        else:
            raise TypeError("embedding vector values must be a tuple or list")
        if not values:
            raise ValueError("embedding vector must not be empty")
        normalized: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError("embedding vector values must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("embedding vector values must be finite")
            normalized.append(number)
        norm = math.sqrt(sum(value * value for value in normalized))
        if norm == 0:
            raise ValueError("embedding vector must have a non-zero norm")
        object.__setattr__(self, "values", tuple(value / norm for value in normalized))

    @property
    def dimension(self) -> int:
        return len(self.values)


class Embedder(Protocol):
    """区分查询与文档输入的异步稠密向量接口。"""

    async def embed_query(self, text: str) -> EmbeddingVector: ...

    async def embed_documents(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]: ...


@dataclass(frozen=True)
class EmbeddingConfig:
    """一条通过 LiteLLM 路由的文本向量配置。"""

    enabled: bool = False
    model: str = ""
    dimension: int = 0
    base_url: str = ""
    api_key_env: str = ""
    timeout_seconds: float = 30.0
    max_batch_size: int = 32
    max_input_chars: int = 16_000
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    extra_body: Mapping[str, object] = field(default_factory=dict)
    query_parameters: Mapping[str, object] = field(default_factory=dict)
    document_parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("embedding enabled must be boolean")
        model = str(self.model or "").strip()
        base_url = str(self.base_url or "").strip().rstrip("/")
        api_key_env = str(self.api_key_env or "").strip()
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "api_key_env", api_key_env)
        if self.enabled and (not model or self.dimension <= 0):
            raise ValueError("enabled embedding config requires model and positive dimension")
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int) or self.dimension < 0:
            raise ValueError("embedding dimension must be a non-negative integer")
        if self.dimension > 65_536:
            raise ValueError("embedding dimension exceeds the supported bound")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int | float)
            or not 0 < float(self.timeout_seconds) <= 600
        ):
            raise ValueError("embedding timeout_seconds must be between zero and 600")
        for name, value, maximum in (
            ("max_batch_size", self.max_batch_size, 2_048),
            ("max_input_chars", self.max_input_chars, 1_000_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"embedding {name} must be between one and {maximum}")
        object.__setattr__(self, "extra_headers", _string_mapping(self.extra_headers, "extra_headers"))
        object.__setattr__(self, "extra_body", _parameter_mapping(self.extra_body, "extra_body"))
        object.__setattr__(
            self,
            "query_parameters",
            _parameter_mapping(self.query_parameters, "query_parameters"),
        )
        object.__setattr__(
            self,
            "document_parameters",
            _parameter_mapping(self.document_parameters, "document_parameters"),
        )
        reserved = {"model", "input", "api_key", "api_base", "timeout"}
        for name in ("extra_body", "query_parameters", "document_parameters"):
            overlap = reserved & set(getattr(self, name))
            if overlap:
                raise ValueError(f"embedding {name} cannot override reserved fields: {sorted(overlap)}")


class LiteLLMEmbedder:
    """通过 LiteLLM 为多个供应商提供同一批量向量接口。"""

    def __init__(self, config: EmbeddingConfig) -> None:
        if not isinstance(config, EmbeddingConfig) or not config.enabled:
            raise ModelConfigurationError("LiteLLMEmbedder requires an enabled EmbeddingConfig")
        self.config = config

    async def embed_query(self, text: str) -> EmbeddingVector:
        normalized = self._text(text, "embedding query")
        return (await self._embed((normalized,), is_query=True))[0]

    async def embed_documents(self, texts: Sequence[str]) -> tuple[EmbeddingVector, ...]:
        if isinstance(texts, str) or not isinstance(texts, Sequence):
            raise TypeError("embedding documents must be a sequence of strings")
        normalized = tuple(self._text(text, "embedding document") for text in texts)
        if not normalized:
            return ()
        result: list[EmbeddingVector] = []
        size = self.config.max_batch_size
        for offset in range(0, len(normalized), size):
            result.extend(await self._embed(normalized[offset : offset + size], is_query=False))
        return tuple(result)

    async def _embed(
        self,
        texts: tuple[str, ...],
        *,
        is_query: bool,
    ) -> tuple[EmbeddingVector, ...]:
        module = self._module()
        call = getattr(module, "aembedding", None)
        if not callable(call):
            raise ModelDependencyError("installed LiteLLM does not provide aembedding")
        async_call = cast(Callable[..., Awaitable[object]], call)
        kwargs: dict[str, object] = {
            "model": self.config.model,
            "input": list(texts),
            "timeout": float(self.config.timeout_seconds),
        }
        api_key = os.environ.get(self.config.api_key_env, "") if self.config.api_key_env else ""
        if api_key:
            kwargs["api_key"] = api_key
        if self.config.base_url:
            kwargs["api_base"] = self.config.base_url
        if self.config.extra_headers:
            kwargs["extra_headers"] = dict(self.config.extra_headers)
        extra_body = dict(self.config.extra_body)
        parameters = self.config.query_parameters if is_query else self.config.document_parameters
        kwargs.update(parameters)
        if extra_body:
            kwargs["extra_body"] = extra_body
        try:
            response = await async_call(**kwargs)
            return self._vectors(response, expected=len(texts))
        except Exception as exc:
            if isinstance(exc, (ModelDependencyError, ModelResponseError)):
                raise
            raise normalize_provider_error(exc) from exc

    def _vectors(self, response: object, *, expected: int) -> tuple[EmbeddingVector, ...]:
        data = _field(response, "data")
        if not isinstance(data, Sequence) or isinstance(data, str) or len(data) != expected:
            raise ModelResponseError("embedding provider returned an unexpected item count")
        indexed: list[tuple[int, EmbeddingVector]] = []
        seen: set[int] = set()
        for fallback_index, item in enumerate(data):
            raw_index = _field(item, "index", fallback_index)
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise ModelResponseError("embedding provider returned an invalid item index")
            if raw_index < 0 or raw_index >= expected or raw_index in seen:
                raise ModelResponseError("embedding provider returned duplicate or out-of-range indexes")
            raw_vector = _field(item, "embedding")
            if not isinstance(raw_vector, Sequence) or isinstance(raw_vector, str):
                raise ModelResponseError("embedding provider returned a malformed vector")
            values = tuple(raw_vector)
            if len(values) < self.config.dimension:
                raise ModelResponseError("embedding vector is shorter than the configured dimension")
            vector = EmbeddingVector(values[: self.config.dimension])
            indexed.append((raw_index, vector))
            seen.add(raw_index)
        indexed.sort(key=lambda item: item[0])
        return tuple(vector for _, vector in indexed)

    def _text(self, value: object, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be non-empty text")
        normalized = value.strip()
        if len(normalized) > self.config.max_input_chars:
            raise ValueError(f"{label} exceeds the configured character bound")
        return normalized

    @staticmethod
    def _module() -> ModuleType:
        try:
            return importlib.import_module("litellm")
        except ImportError as exc:
            raise ModelDependencyError("embedding requires the optional 'litellm' package") from exc


def build_embedder(config: EmbeddingConfig) -> LiteLLMEmbedder:
    """构造启用的默认 Embedder；禁用配置不能静默退化。"""

    return LiteLLMEmbedder(config)


def _field(source: object, name: str, default: object = None) -> object:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _string_mapping(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"embedding {label} must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(item, str):
            raise ValueError(f"embedding {label} must contain non-empty string keys and string values")
        result[key] = item
    return result


def _parameter_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"embedding {label} must be an object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"embedding {label} keys must be non-empty strings")
        result[key] = item
    return result


__all__ = [
    "Embedder",
    "EmbeddingConfig",
    "EmbeddingVector",
    "LiteLLMEmbedder",
    "build_embedder",
]
