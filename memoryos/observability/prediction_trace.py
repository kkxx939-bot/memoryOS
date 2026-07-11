"""日志追踪里的预测追踪。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PredictionTrace:
    episode_id: str
    top_action: str
    candidate_count: int
    feature_summary: dict[str, float] = field(default_factory=dict)
    policy_decision: dict = field(default_factory=dict)
    model_versions: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "top_action": self.top_action,
            "candidate_count": self.candidate_count,
            "feature_summary": self.feature_summary,
            "policy_decision": self.policy_decision,
            "model_versions": self.model_versions,
        }
