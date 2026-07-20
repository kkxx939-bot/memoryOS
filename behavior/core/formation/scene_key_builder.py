"""行为模块里的场景标识组装器。"""

from __future__ import annotations

from behavior.core.model.observation import Observation


class SceneKeyBuilder:
    def build(self, observation: Observation) -> str:
        return observation.scene_key
