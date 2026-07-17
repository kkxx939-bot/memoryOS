"""Reranking capability consumed by context retrieval orchestration."""

from __future__ import annotations

from typing import Any, Protocol


class Reranker(Protocol):
    def rerank(self, query: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


__all__ = ["Reranker"]
