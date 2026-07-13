"""上下文数据库里的向量存储。"""

from __future__ import annotations

import math
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

    def get_vector_metadata(self, uri: str) -> dict | None: ...

    def vector_uris(self) -> list[str]: ...


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.rows: dict[str, tuple[list[float], dict]] = {}

    def upsert_vector(self, uri: str, embedding: list[float], metadata: dict | None = None) -> None:
        self.rows[uri] = (_finite_vector(embedding), metadata or {})

    def delete_vector(self, uri: str) -> None:
        self.rows.pop(uri, None)

    def get_vector_metadata(self, uri: str) -> dict | None:
        row = self.rows.get(uri)
        return dict(row[1]) if row is not None else None

    def vector_uris(self) -> list[str]:
        return list(self.rows)

    def search_vector(self, embedding: list[float], namespace: str, limit: int = 10) -> list[VectorHit]:
        embedding = _finite_vector(embedding)
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


def _finite_vector(values: list[float]) -> list[float]:
    result = [float(value) for value in values]
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError("vector values must be finite and non-empty")
    return result
