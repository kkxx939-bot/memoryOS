from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OpportunityStats:
    opportunity_count: int = 0
    activation_count: int = 0
    missed_opportunity_count: int = 0
    negative_feedback_count: int = 0
    last_opportunity_at: str | None = None
    last_activated_at: str | None = None


@dataclass(frozen=True)
class OpportunityDecayResult:
    opportunity_state: str
    hotness_delta: float
    q_value_delta: float
    reason: str
    generated_operations: list | None = None

    def __post_init__(self) -> None:
        if self.generated_operations is None:
            object.__setattr__(self, "generated_operations", [])
