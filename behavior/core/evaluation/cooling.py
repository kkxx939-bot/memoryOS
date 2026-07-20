"""行为模块里的行为冷却。"""

from __future__ import annotations

from behavior.core.evaluation.opportunity_decay import OpportunityAwareDecay
from behavior.core.model.behavior_pattern import BehaviorPattern
from behavior.core.model.observation import Observation


class BehaviorCoolingService:
    def cool(self, pattern: BehaviorPattern, recent_observations: list[Observation]):
        return OpportunityAwareDecay().evaluate(pattern, recent_observations)
