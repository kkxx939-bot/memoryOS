from __future__ import annotations

from typing import Any, Protocol


class Reranker(Protocol):
    def rerank(self, query: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


class NoopReranker:
    def rerank(self, query: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:  # noqa: ARG002
        return items
