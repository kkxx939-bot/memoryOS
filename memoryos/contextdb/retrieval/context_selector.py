"""上下文数据库里的上下文筛选。"""

from __future__ import annotations

from memoryos.contextdb.store.index_store import IndexHit


class ContextSelector:
    def select(self, hits: list[IndexHit], limit: int) -> list[IndexHit]:
        seen = set()
        selected = []
        for hit in sorted(hits, key=lambda item: item.score, reverse=True):
            if hit.uri in seen:
                continue
            seen.add(hit.uri)
            selected.append(hit)
            if len(selected) >= limit:
                break
        return selected
