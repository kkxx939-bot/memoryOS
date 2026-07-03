from __future__ import annotations

from memoryos.domain.scene.observation import ObservationContext
from memoryos.domain.scene.scene_features import SceneFeatures


def context_tags_from_observation(observation: ObservationContext) -> list[str]:
    return observation.context_tags()


def context_tags_from_features(features: SceneFeatures) -> list[str]:
    tags = []
    for value in [features.location, features.activity, features.time_bucket, features.thermal_level]:
        if value:
            tags.append(value)
    tags.extend(features.signals)
    return sorted({str(tag) for tag in tags if str(tag).strip()})
