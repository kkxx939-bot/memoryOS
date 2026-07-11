"""检索结果重排。"""

from __future__ import annotations

from memoryos.contextdb.store.source_store import IndexHit


class ContextReranker:
    def rerank(self, hits: list[IndexHit]) -> list[IndexHit]:
        return sorted(hits, key=lambda item: item.score, reverse=True)
