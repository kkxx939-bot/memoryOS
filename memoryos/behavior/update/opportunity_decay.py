from __future__ import annotations

from memoryos.behavior.model.behavior_pattern import BehaviorPattern
from memoryos.behavior.model.observation import Observation
from memoryos.behavior.model.opportunity import OpportunityDecayResult
from memoryos.behavior.update.opportunity_window import build_opportunity_window


class OpportunityAwareDecay:
    def evaluate(self, pattern: BehaviorPattern, recent_observations: list[Observation]) -> OpportunityDecayResult:
        opportunities = [obs for obs in recent_observations if self._matches(pattern, obs)]
        window = build_opportunity_window(opportunities)
        if not opportunities:
            return OpportunityDecayResult(
                opportunity_state="no_opportunity",
                hotness_delta=-0.005,
                q_value_delta=0.0,
                reason="No matching trigger opportunity occurred.",
                recent_opportunity_count=0,
                window_start=window.window_start,
                window_end=window.window_end,
            )
        if window.negative_count > 0:
            return OpportunityDecayResult(
                opportunity_state="negative_feedback",
                hotness_delta=-0.20,
                q_value_delta=-0.25,
                reason="Recent trigger opportunities include negative feedback.",
                recent_opportunity_count=window.opportunity_count,
                recent_activation_count=window.activation_count,
                recent_missed_count=window.missed_count,
                recent_negative_count=window.negative_count,
                window_start=window.window_start,
                window_end=window.window_end,
            )
        missed = window.missed_count
        activated = window.activation_count
        if activated >= missed:
            return OpportunityDecayResult(
                opportunity_state="opportunity_activated",
                hotness_delta=0.08,
                q_value_delta=0.06,
                reason="Matching opportunities still activate the behavior.",
                recent_opportunity_count=window.opportunity_count,
                recent_activation_count=window.activation_count,
                recent_missed_count=window.missed_count,
                recent_negative_count=window.negative_count,
                window_start=window.window_start,
                window_end=window.window_end,
            )
        return OpportunityDecayResult(
            opportunity_state="opportunity_missed",
            hotness_delta=-0.06,
            q_value_delta=-0.08,
            reason="Matching opportunities occurred without behavior activation.",
            recent_opportunity_count=window.opportunity_count,
            recent_activation_count=window.activation_count,
            recent_missed_count=window.missed_count,
            recent_negative_count=window.negative_count,
            window_start=window.window_start,
            window_end=window.window_end,
        )

    def _matches(self, pattern: BehaviorPattern, observation: Observation) -> bool:
        conditions = pattern.trigger_conditions or {}
        if conditions.get("scene_key") and conditions["scene_key"] != observation.scene_key:
            return False
        tags = set(observation.context_tags())
        required = set(conditions.get("context_tags", []))
        if required and not required.issubset(tags):
            return False
        if conditions.get("location") and str(conditions["location"]) != observation.location:
            return False
        if conditions.get("activity") and str(conditions["activity"]) != observation.activity:
            return False
        environment = conditions.get("environment", {}) or {}
        if isinstance(environment, dict) and not self._environment_matches(environment, observation.environment):
            return False
        if any(key in conditions for key in ("temperature_gte", "temperature_lte")):
            if not self._environment_matches(conditions, observation.environment):
                return False
        return bool(conditions.get("scene_key") or required or conditions.get("location") or conditions.get("activity") or environment)

    def _environment_matches(self, conditions: dict, environment: dict) -> bool:
        temperature = environment.get("temperature")
        try:
            gte = conditions.get("temperature_gte")
            lte = conditions.get("temperature_lte")
            if gte is None and lte is None:
                return True
            if (gte is not None or lte is not None) and temperature is None:
                return False
            assert temperature is not None
            observed = float(temperature)
            if gte is not None and observed < float(gte):
                return False
            if lte is not None and observed > float(lte):
                return False
        except (TypeError, ValueError):
            return False
        return True
