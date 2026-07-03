from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class HybridHit:
    id: str
    score: float
    source_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    text: str = ""


class HybridIndex(Protocol):
    backend_name: str

    def search(
        self,
        *,
        namespace: str,
        query: str,
        query_vector: list[float] | None = None,
        top_k: int = 8,
        filters: dict | None = None,
    ) -> list[HybridHit]: ...

    def health_check(self) -> dict: ...
