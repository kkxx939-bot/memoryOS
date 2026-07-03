from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetrievalTrace:
    query: str
    route: str
    candidate_count: int
    selected_count: int
    source_scores: dict[str, float] = field(default_factory=dict)
    provider_metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "route": self.route,
            "candidate_count": self.candidate_count,
            "selected_count": self.selected_count,
            "source_scores": self.source_scores,
            "provider_metadata": self.provider_metadata,
        }
