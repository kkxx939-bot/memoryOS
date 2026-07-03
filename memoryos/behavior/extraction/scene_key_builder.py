from __future__ import annotations

from memoryos.behavior.model.observation import Observation


class SceneKeyBuilder:
    def build(self, observation: Observation) -> str:
        return observation.scene_key
