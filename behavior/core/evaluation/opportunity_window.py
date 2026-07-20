"""行为模块里的机会窗口。"""

from __future__ import annotations

from dataclasses import dataclass

from behavior.core.model.observation import Observation


@dataclass(frozen=True)
class OpportunityWindow:
    observations: list[Observation]
    opportunity_count: int
    activation_count: int
    missed_count: int
    negative_count: int
    window_start: str | None
    window_end: str | None


def build_opportunity_window(observations: list[Observation]) -> OpportunityWindow:
    activated = 0
    missed = 0
    negative = 0
    timestamps = [observation.observed_at for observation in observations if observation.observed_at]
    for observation in observations:
        signals = set(observation.signals)
        if signals.intersection({"negative_feedback", "user_rejected", "implicit_negative", "explicit_negative_rule"}):
            negative += 1
        if signals.intersection({"behavior_activated", "action_executed", "positive_feedback", "implicit_positive"}):
            activated += 1
        elif signals.intersection({"missed_opportunity", "not_activated", "ignored"}):
            missed += 1
    if observations and activated == 0 and missed == 0 and negative == 0:
        missed = len(observations)
    return OpportunityWindow(
        observations=observations,
        opportunity_count=len(observations),
        activation_count=activated,
        missed_count=missed,
        negative_count=negative,
        window_start=min(timestamps) if timestamps else None,
        window_end=max(timestamps) if timestamps else None,
    )
