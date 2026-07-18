from __future__ import annotations

from memoryos.behavior.model.behavior_pattern import BehaviorPattern
from memoryos.behavior.model.observation import Observation
from memoryos.behavior.model.opportunity import OpportunityStats
from memoryos.behavior.update.opportunity_decay import OpportunityAwareDecay


def _pattern(stats: OpportunityStats | None = None) -> BehaviorPattern:
    return BehaviorPattern(
        user_id="u1",
        scene_key="hot_room",
        trigger_conditions={"context_tags": ["home", "hot_environment"]},
        support_anchor_uri="memoryos://user/u1/support/behavior/hot",
        case_refs=["c1", "c2"],
        action_distribution=[{"action": "turn_on_ac", "count": 2}],
        opportunity=stats or OpportunityStats(),
    )


def test_no_recent_opportunity_does_not_penalize_q_value() -> None:
    result = OpportunityAwareDecay().evaluate(_pattern(), [Observation(user_id="u1", location="office")])
    assert result.opportunity_state == "no_opportunity"
    assert result.q_value_delta == 0.0


def test_activated_opportunity_rewards_or_refreshes_behavior() -> None:
    result = OpportunityAwareDecay().evaluate(
        _pattern(OpportunityStats(activation_count=2, missed_opportunity_count=1)),
        [Observation(user_id="u1", location="home", signals=["action_executed"], environment={"temperature": 30})],
    )
    assert result.opportunity_state == "opportunity_activated"
    assert result.q_value_delta > 0


def test_missed_opportunity_lightly_penalizes_policy() -> None:
    result = OpportunityAwareDecay().evaluate(
        _pattern(OpportunityStats(activation_count=0, missed_opportunity_count=2)),
        [Observation(user_id="u1", location="home", environment={"temperature": 30})],
    )
    assert result.opportunity_state == "opportunity_missed"
    assert -0.1 <= result.q_value_delta < 0


def test_negative_feedback_has_stronger_penalty() -> None:
    result = OpportunityAwareDecay().evaluate(
        _pattern(OpportunityStats(negative_feedback_count=1)),
        [Observation(user_id="u1", location="home", signals=["negative_feedback"], environment={"temperature": 30})],
    )
    assert result.opportunity_state == "negative_feedback"
    assert result.q_value_delta < -0.1
