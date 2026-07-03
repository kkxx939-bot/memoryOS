from __future__ import annotations

import hashlib
import json

from memoryos.domain.scene.scene_features import SceneFeatures


def stable_scene_signature(features: SceneFeatures) -> str:
    material = {
        "location": features.location,
        "activity": features.activity,
        "time_bucket": features.time_bucket,
        "thermal_level": features.thermal_level,
        "signals": sorted(features.signals),
        "environment_bucket": _environment_bucket(features.environment),
    }
    digest = hashlib.sha256(json.dumps(material, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"scene_{digest}"


def _environment_bucket(environment: dict) -> dict:
    bucket: dict[str, object] = {}
    for key, value in environment.items():
        if isinstance(value, (int, float)):
            bucket[str(key)] = round(float(value))
        else:
            bucket[str(key)] = str(value)
    return bucket
