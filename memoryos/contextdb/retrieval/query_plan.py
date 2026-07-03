from __future__ import annotations

from dataclasses import dataclass, field

from memoryos.contextdb.model.context_type import ContextType


@dataclass(frozen=True)
class QueryPlan:
    query: str
    user_id: str
    context_types: list[ContextType]
    token_budget: int
    steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "user_id": self.user_id,
            "context_types": [item.value for item in self.context_types],
            "token_budget": self.token_budget,
            "steps": self.steps,
        }
