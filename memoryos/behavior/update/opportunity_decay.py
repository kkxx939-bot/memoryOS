from __future__ import annotations

from memoryos.behavior.model.behavior_pattern import BehaviorPattern
from memoryos.behavior.model.observation import Observation
from memoryos.behavior.model.opportunity import OpportunityDecayResult


class OpportunityAwareDecay:
    def evaluate(self, pattern: BehaviorPattern, recent_observations: list[Observation]) -> OpportunityDecayResult:
        opportunities = [obs for obs in recent_observations if self._matches(pattern, obs)]
        if not opportunities:
            return OpportunityDecayResult(
                opportunity_state="no_opportunity",
                hotness_delta=-0.005,
                q_value_delta=0.0,
                reason="No matching trigger opportunity occurred.",
            )
        if pattern.opportunity.negative_feedback_count > 0:
            return OpportunityDecayResult(
                opportunity_state="negative_feedback",
                hotness_delta=-0.20,
                q_value_delta=-0.25,
                reason="Recent trigger opportunities include negative feedback.",
            )
        missed = pattern.opportunity.missed_opportunity_count
        activated = pattern.opportunity.activation_count
        if activated >= missed:
            return OpportunityDecayResult(
                opportunity_state="opportunity_activated",
                hotness_delta=0.08,
                q_value_delta=0.06,
                reason="Matching opportunities still activate the behavior.",
            )
        return OpportunityDecayResult(
            opportunity_state="opportunity_missed",
            hotness_delta=-0.06,
            q_value_delta=-0.08,
            reason="Matching opportunities occurred without behavior activation.",
        )

    def _matches(self, pattern: BehaviorPattern, observation: Observation) -> bool:
        conditions = pattern.trigger_conditions or {}
        if conditions.get("scene_key") and conditions["scene_key"] == observation.scene_key:
            return True
        tags = set(observation.context_tags())
        required = set(conditions.get("context_tags", []))
        return bool(required and required.issubset(tags))
