from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class RerankDocument:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RerankHit:
    id: str
    score: float
    model: str = ""
    provider: str = ""
    reason: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "score": self.score,
            "model": self.model,
            "provider": self.provider,
            "reason": self.reason,
            "metadata": self.metadata,
        }


class RerankProvider(Protocol):
    provider_name: str
    model: str

    def rerank(self, query: str, documents: list[str] | list[RerankDocument]) -> list[float] | list[RerankHit] | None:
        """Return one relevance score/hit per document, in input order."""

    def health_check(self) -> dict: ...


def rerank_with_fallback(
    provider: RerankProvider | None,
    query: str,
    documents: list[str],
    fallback_scores: list[float],
) -> list[float]:
    if provider is None or not documents:
        return fallback_scores
    try:
        scores = provider.rerank(query, documents)
    except Exception:
        return fallback_scores
    if not scores or len(scores) != len(documents):
        return fallback_scores
    if all(isinstance(score, RerankHit) for score in scores):
        return [
            _finite_score(score.score, fallback)
            for score, fallback in zip(scores, fallback_scores, strict=True)
            if isinstance(score, RerankHit)
        ]
    numeric_scores = [score for score in scores if not isinstance(score, RerankHit)]
    return [_finite_score(score, fallback) for score, fallback in zip(numeric_scores, fallback_scores, strict=True)]


def _finite_score(value: object, fallback: float) -> float:
    if not isinstance(value, int | float | str):
        return fallback
    try:
        score = float(value)
    except (TypeError, ValueError):
        return fallback
    if score != score or score in {float("inf"), float("-inf")}:
        return fallback
    return max(0.0, min(1.0, score))
