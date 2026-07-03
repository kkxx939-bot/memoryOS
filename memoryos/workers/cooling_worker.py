from __future__ import annotations

from memoryos.behavior.model.behavior_pattern import BehaviorPattern
from memoryos.behavior.model.observation import Observation
from memoryos.behavior.update.behavior_cooling import BehaviorCoolingService


class CoolingWorker:
    def process_behavior_patterns(self, patterns: list[BehaviorPattern], observations: list[Observation]) -> list[dict]:
        results = []
        for pattern in patterns:
            result = BehaviorCoolingService().cool(pattern, observations)
            results.append(result.__dict__)
        return results
