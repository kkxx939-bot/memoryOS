"""上下文检索流水线依赖的重排能力协议。"""

from __future__ import annotations

from typing import Any, Protocol


class Reranker(Protocol):
    def rerank(self, query: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


__all__ = ["Reranker"]
