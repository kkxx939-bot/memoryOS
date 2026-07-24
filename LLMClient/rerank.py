"""供应商无关的文本重排契约与 LiteLLM 适配器。"""

from __future__ import annotations

import asyncio
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


class Reranker(Protocol):
    """按照输入原顺序返回相关性分数的异步重排接口。"""

    async def rerank(self, query: str, documents: Sequence[str]) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class RerankConfig:
    """一条通过 LiteLLM 路由的重排模型配置。"""

    enabled: bool = False
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""
    timeout_seconds: float = 30.0
    max_documents: int = 100
    max_query_chars: int = 8_000
    max_document_chars: int = 16_000
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    extra_body: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("rerank enabled must be boolean")
        model = str(self.model or "").strip()
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "base_url", str(self.base_url or "").strip().rstrip("/"))
        object.__setattr__(self, "api_key_env", str(self.api_key_env or "").strip())
        if self.enabled and not model:
            raise ValueError("enabled rerank config requires model")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int | float)
            or not 0 < float(self.timeout_seconds) <= 600
        ):
            raise ValueError("rerank timeout_seconds must be between zero and 600")
        for name, value, maximum in (
            ("max_documents", self.max_documents, 2_048),
            ("max_query_chars", self.max_query_chars, 1_000_000),
            ("max_document_chars", self.max_document_chars, 1_000_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"rerank {name} must be between one and {maximum}")
        object.__setattr__(self, "extra_headers", _string_mapping(self.extra_headers))
        if not isinstance(self.extra_body, Mapping):
            raise TypeError("rerank extra_body must be an object")
        extra_body = dict(self.extra_body)
        overlap = {"model", "query", "documents", "api_key", "api_base", "timeout"} & set(extra_body)
        if overlap:
            raise ValueError(f"rerank extra_body cannot override reserved fields: {sorted(overlap)}")
        object.__setattr__(self, "extra_body", extra_body)


class LiteLLMReranker:
    """通过 LiteLLM 的统一 rerank API 调用具体供应商。"""

    def __init__(self, config: RerankConfig) -> None:
        if not isinstance(config, RerankConfig) or not config.enabled:
            raise ModelConfigurationError("LiteLLMReranker requires an enabled RerankConfig")
        self.config = config

    async def rerank(self, query: str, documents: Sequence[str]) -> tuple[float, ...]:
        normalized_query = self._text(query, self.config.max_query_chars, "rerank query")
        if isinstance(documents, str) or not isinstance(documents, Sequence):
            raise TypeError("rerank documents must be a sequence of strings")
        normalized_documents = tuple(
            self._text(document, self.config.max_document_chars, "rerank document")
            for document in documents
        )
        if not normalized_documents:
            return ()
        if len(normalized_documents) > self.config.max_documents:
            raise ValueError("rerank document count exceeds the configured bound")

        module = self._module()
        kwargs: dict[str, object] = {
            "model": self.config.model,
            "query": normalized_query,
            "documents": list(normalized_documents),
            "timeout": float(self.config.timeout_seconds),
            **dict(self.config.extra_body),
        }
        api_key = os.environ.get(self.config.api_key_env, "") if self.config.api_key_env else ""
        if api_key:
            kwargs["api_key"] = api_key
        if self.config.base_url:
            kwargs["api_base"] = self.config.base_url
        if self.config.extra_headers:
            kwargs["extra_headers"] = dict(self.config.extra_headers)
        try:
            async_call = getattr(module, "arerank", None)
            if callable(async_call):
                response = await cast(Callable[..., Awaitable[object]], async_call)(**kwargs)
            else:
                sync_call = getattr(module, "rerank", None)
                if not callable(sync_call):
                    raise ModelDependencyError("installed LiteLLM does not provide rerank")
                response = await asyncio.to_thread(sync_call, **kwargs)
            return self._scores(response, expected=len(normalized_documents))
        except Exception as exc:
            if isinstance(exc, (ModelDependencyError, ModelResponseError)):
                raise
            raise normalize_provider_error(exc) from exc

    @staticmethod
    def _scores(response: object, *, expected: int) -> tuple[float, ...]:
        results = _field(response, "results")
        if not isinstance(results, Sequence) or isinstance(results, str) or len(results) != expected:
            raise ModelResponseError("rerank provider returned an unexpected result count")
        scores: list[float | None] = [None] * expected
        for fallback_index, item in enumerate(results):
            raw_index = _field(item, "index", fallback_index)
            raw_score = _field(item, "relevance_score", _field(item, "score"))
            if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                raise ModelResponseError("rerank provider returned an invalid result index")
            if raw_index < 0 or raw_index >= expected or scores[raw_index] is not None:
                raise ModelResponseError("rerank provider returned duplicate or out-of-range indexes")
            if isinstance(raw_score, bool) or not isinstance(raw_score, int | float):
                raise ModelResponseError("rerank provider returned a non-numeric score")
            score = float(raw_score)
            if not math.isfinite(score):
                raise ModelResponseError("rerank provider returned a non-finite score")
            scores[raw_index] = score
        if any(score is None for score in scores):
            raise ModelResponseError("rerank provider omitted a document score")
        return tuple(float(score) for score in scores if score is not None)

    @staticmethod
    def _text(value: object, maximum: int, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be non-empty text")
        normalized = value.strip()
        if len(normalized) > maximum:
            raise ValueError(f"{label} exceeds the configured character bound")
        return normalized

    @staticmethod
    def _module() -> ModuleType:
        try:
            return importlib.import_module("litellm")
        except ImportError as exc:
            raise ModelDependencyError("rerank requires the optional 'litellm' package") from exc


def build_reranker(config: RerankConfig) -> LiteLLMReranker:
    """构造启用的默认 Reranker；禁用配置由上层显式表示为 None。"""

    return LiteLLMReranker(config)


def _field(source: object, name: str, default: object = None) -> object:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _string_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("rerank extra_headers must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(item, str):
            raise ValueError("rerank extra_headers must contain non-empty string keys and string values")
        result[key] = item
    return result


__all__ = [
    "LiteLLMReranker",
    "RerankConfig",
    "Reranker",
    "build_reranker",
]
