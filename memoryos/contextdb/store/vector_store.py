"""上下文数据库里的向量存储。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class VectorHit:
    uri: str
    score: float
    metadata: dict = field(default_factory=dict)


class VectorStore(Protocol):
    def upsert_vector(self, uri: str, embedding: list[float], metadata: dict | None = None) -> None: ...

    def delete_vector(self, uri: str) -> None: ...

    def search_vector(self, embedding: list[float], namespace: str, limit: int = 10) -> list[VectorHit]: ...


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.rows: dict[str, tuple[list[float], dict]] = {}

    def upsert_vector(self, uri: str, embedding: list[float], metadata: dict | None = None) -> None:
        self.rows[uri] = (list(embedding), metadata or {})

    def delete_vector(self, uri: str) -> None:
        self.rows.pop(uri, None)

    def search_vector(self, embedding: list[float], namespace: str, limit: int = 10) -> list[VectorHit]:
        hits = []
        for uri, (stored, metadata) in self.rows.items():
            if namespace and not uri.startswith(namespace):
                continue
            score = self._cosine(embedding, stored)
            hits.append(VectorHit(uri=uri, score=score, metadata=metadata))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:limit]

    def _cosine(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = sum(a * a for a in left) ** 0.5
        right_norm = sum(b * b for b in right) ** 0.5
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return max(0.0, min(1.0, dot / (left_norm * right_norm)))
