"""行为模块里的行为提取器。"""

from __future__ import annotations

from memoryos.behavior.model.behavior_case import BehaviorCase
from memoryos.behavior.model.observation import Observation


class BehaviorExtractor:
    def extract_case(
        self,
        observation: Observation,
        predicted_candidates: list[dict] | None = None,
        selected_action: str | None = None,
    ) -> BehaviorCase:
        return BehaviorCase(
            user_id=observation.user_id,
            scene_key=observation.scene_key,
            observation=observation.__dict__,
            predicted_candidates=predicted_candidates or [],
            selected_action=selected_action,
        )
