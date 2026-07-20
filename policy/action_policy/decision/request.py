"""ActionPolicy 在线决策请求。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from behavior.core.model.observation import Observation
from policy.action_policy.risk import canonical_action


@dataclass(frozen=True)
class PredictionRequest:
    user_id: str
    episode_id: str
    observation: Observation | dict | str
    available_actions: list[str]
    request_id: str = ""
    snapshot_version: str = ""
    session_uri: str = ""
    resources: list[dict] = field(default_factory=list)
    skills: list[dict] = field(default_factory=list)
    connect_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.user_id).strip():
            raise ValueError("PredictionRequest requires user_id")
        if not str(self.episode_id).strip():
            raise ValueError("PredictionRequest requires episode_id")
        actions = [canonical_action(action) for action in self.available_actions]
        object.__setattr__(
            self,
            "available_actions",
            list(dict.fromkeys(action for action in actions if action)),
        )
