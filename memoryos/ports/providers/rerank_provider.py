from __future__ import annotations

from typing import Protocol


class RerankProvider(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[float] | None:
        """Return one relevance score per document, in input order."""


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
    return [_finite_score(score, fallback) for score, fallback in zip(scores, fallback_scores, strict=True)]


def _finite_score(value: float, fallback: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return fallback
    if score != score or score in {float("inf"), float("-inf")}:
        return fallback
    return max(0.0, min(1.0, score))
