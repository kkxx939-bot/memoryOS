from __future__ import annotations

from dataclasses import dataclass, field

from memoryos.core.ids import new_id
from memoryos.core.time import utc_now


@dataclass
class BehaviorCase:
    user_id: str
    scene_key: str
    observation: dict
    case_id: str = field(default_factory=lambda: new_id("case"))
    predicted_candidates: list[dict] = field(default_factory=list)
    selected_action: str | None = None
    executed_action: str | None = None
    user_actual_action: str | None = None
    feedback_type: str = "unknown"
    reward: float = 0.0
    related_memory_uris: list[str] = field(default_factory=list)
    related_policy_uris: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict:
        observed_at = str(self.observation.get("observed_at") or self.created_at)
        return {
            "case_id": self.case_id,
            "user_id": self.user_id,
            "scene_key": self.scene_key,
            "observation": self.observation,
            "observed_at": observed_at,
            "predicted_candidates": self.predicted_candidates,
            "selected_action": self.selected_action,
            "executed_action": self.executed_action,
            "user_actual_action": self.user_actual_action,
            "feedback_type": self.feedback_type,
            "reward": self.reward,
            "related_memory_uris": self.related_memory_uris,
            "related_policy_uris": self.related_policy_uris,
            "created_at": self.created_at,
        }
