from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetrievalEvalCase:
    case_id: str
    query: str
    expected_ids: list[str]
    context_tags: list[str] = field(default_factory=list)


def precision_at_k(results: list[str], expected: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = results[:k]
    if not top:
        return 0.0
    expected_set = set(expected)
    return len([item for item in top if item in expected_set]) / len(top)


def recall_at_k(results: list[str], expected: list[str], k: int) -> float:
    expected_set = set(expected)
    if not expected_set:
        return 0.0
    top = set(results[:k])
    return len(top & expected_set) / len(expected_set)
