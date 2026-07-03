from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class VectorRecord:
    namespace: str
    id: str
    vector: list[float]
    text: str
    metadata: dict = field(default_factory=dict)
    content_hash: str = ""
    provider: str = ""
    model: str = ""
    dimension: int = 0
    embedding_version: str = "v1"


@dataclass(frozen=True)
class VectorHit:
    id: str
    score: float
    metadata: dict = field(default_factory=dict)
    text: str = ""
    namespace: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "score": self.score,
            "metadata": self.metadata,
            "text": self.text,
            "namespace": self.namespace,
        }


class VectorIndex(Protocol):
    backend_name: str

    def upsert(self, record: VectorRecord) -> None: ...

    def delete(self, *, namespace: str, id: str) -> None: ...

    def search(
        self,
        *,
        namespace: str,
        query_vector: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[VectorHit]: ...

    def health_check(self) -> dict: ...
