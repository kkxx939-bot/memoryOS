"""上下文向量召回依赖的嵌入能力协议。"""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    """向量召回后端必须实现的最小能力契约。"""

    model_name: str
    dimension: int

    def embed(self, text: str) -> list[float]: ...


__all__ = ["EmbeddingProvider"]
