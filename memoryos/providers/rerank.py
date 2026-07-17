"""模型服务的结果重排接口。"""

from __future__ import annotations

from typing import Any

from memoryos.application.context.reranking import Reranker


class NoopReranker:
    def rerank(self, query: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:  # noqa: ARG002
        return items


__all__ = ["NoopReranker", "Reranker"]
