"""把外部观察统一为行为领域的 Observation。"""

from __future__ import annotations

from behavior.core.model.observation import Observation


class ObservationNormalizer:
    def normalize(self, user_id: str, observation: Observation | dict | str) -> Observation:
        if isinstance(observation, Observation):
            if observation.user_id != user_id:
                raise ValueError("observation owner does not match request user")
            return observation
        if isinstance(observation, dict):
            raw_scene_key = observation.get("scene_key")
            return Observation(
                user_id=user_id,
                raw_text=str(observation.get("raw_text", observation.get("scene", ""))),
                location=str(observation.get("location", "")),
                activity=str(observation.get("activity", "")),
                signals=[str(item) for item in observation.get("signals", [])],
                environment=dict(observation.get("environment", {})),
                observed_at=str(observation.get("observed_at", "")),
                explicit_scene_key=str(raw_scene_key) if raw_scene_key is not None else "",
            )
        return Observation(user_id=user_id, raw_text=str(observation))
