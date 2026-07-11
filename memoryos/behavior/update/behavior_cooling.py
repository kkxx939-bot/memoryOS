"""行为模块里的行为冷却。"""

from __future__ import annotations

from memoryos.behavior.model.behavior_pattern import BehaviorPattern
from memoryos.behavior.model.observation import Observation
from memoryos.behavior.update.opportunity_decay import OpportunityAwareDecay


class BehaviorCoolingService:
    def cool(self, pattern: BehaviorPattern, recent_observations: list[Observation]):
        return OpportunityAwareDecay().evaluate(pattern, recent_observations)
