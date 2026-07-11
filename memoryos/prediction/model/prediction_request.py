"""预测模块里的预测请求。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memoryos.behavior.model.observation import Observation


@dataclass(frozen=True)
class PredictionRequest:
    user_id: str
    episode_id: str
    observation: Observation | dict | str
    available_actions: list[str]
    token_budget: int = 2000
    request_id: str = ""
    snapshot_version: str = ""
    session_uri: str = ""
    resources: list[dict] = field(default_factory=list)
    skills: list[dict] = field(default_factory=list)
    connect_metadata: dict[str, Any] = field(default_factory=dict)
