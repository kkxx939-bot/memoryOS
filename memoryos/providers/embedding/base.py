"""向量服务的基础接口。"""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    model_name: str
    dimension: int

    def embed(self, text: str) -> list[float]: ...


class NoopEmbeddingProvider:
    model_name = "noop"
    dimension = 0

    def embed(self, text: str) -> list[float]:  # noqa: ARG002
        return []
