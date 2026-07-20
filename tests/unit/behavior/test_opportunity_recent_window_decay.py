from __future__ import annotations

from behavior.core.evaluation.opportunity_decay import OpportunityAwareDecay
from behavior.core.model.behavior_pattern import BehaviorPattern
from behavior.core.model.observation import Observation
from behavior.core.model.opportunity import OpportunityStats
from behavior.execute.cooling_worker import CoolingWorker
from behavior.projection import behavior_pattern_to_context_object
from infrastructure.context.operation_effects import InfrastructureContextOperationEffects
from tests.support.persistence import FileSystemSourceStore, InMemoryIndexStore
from tests.support.transaction import build_test_operation_committer as OperationCommitter


def _pattern(stats: OpportunityStats | None = None, trigger_conditions: dict | None = None) -> BehaviorPattern:
    return BehaviorPattern(
        user_id="u1",
        scene_key="hot_room",
        trigger_conditions=trigger_conditions or {"context_tags": ["home", "hot_environment"]},
        support_anchor_uri="memoryos://user/u1/support/behavior/hot",
        case_refs=["c1", "c2", "c3"],
        action_distribution=[{"action": "turn_on_ac", "count": 3}],
        opportunity=stats or OpportunityStats(),
    )


def test_recent_no_opportunity_ignores_old_negative_feedback_and_no_penalty(tmp_path) -> None:
    pattern = _pattern(OpportunityStats(negative_feedback_count=99))
    result = OpportunityAwareDecay().evaluate(pattern, [Observation(user_id="u1", location="office")])
    assert result.opportunity_state == "no_opportunity"
    assert result.q_value_delta == 0.0

    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    source.write_object(behavior_pattern_to_context_object(pattern), content="hot room")
    index.upsert_index(behavior_pattern_to_context_object(pattern), content="hot room", tenant_id="default")
    worker_result = CoolingWorker(
        source,
        index,
        OperationCommitter(
            source,
            index,
            str(tmp_path),
            context_effects=InfrastructureContextOperationEffects(),
        ),
    ).process_behavior_patterns(
        "u1", [Observation(user_id="u1", raw_text="hot room", location="office")]
    )
    assert worker_result["operations"] == []


def test_recent_activated_opportunity_strengthens() -> None:
    result = OpportunityAwareDecay().evaluate(
        _pattern(),
        [Observation(user_id="u1", location="home", signals=["action_executed"], environment={"temperature": 30})],
    )
    assert result.opportunity_state == "opportunity_activated"
    assert result.recent_activation_count == 1
    assert result.hotness_delta > 0
    assert result.q_value_delta > 0


def test_recent_missed_opportunity_lightly_decays() -> None:
    result = OpportunityAwareDecay().evaluate(
        _pattern(),
        [Observation(user_id="u1", location="home", signals=["missed_opportunity"], environment={"temperature": 30})],
    )
    assert result.opportunity_state == "opportunity_missed"
    assert result.recent_missed_count == 1
    assert -0.1 <= result.q_value_delta < 0


def test_recent_negative_feedback_strongly_decays() -> None:
    result = OpportunityAwareDecay().evaluate(
        _pattern(),
        [Observation(user_id="u1", location="home", signals=["negative_feedback"], environment={"temperature": 30})],
    )
    assert result.opportunity_state == "negative_feedback"
    assert result.recent_negative_count == 1
    assert result.q_value_delta < -0.1


def test_temperature_trigger_condition_matches_hot_weather() -> None:
    result = OpportunityAwareDecay().evaluate(
        _pattern(trigger_conditions={"location": "home", "environment": {"temperature_gte": 29}}),
        [Observation(user_id="u1", location="home", signals=["action_executed"], environment={"temperature": 30})],
    )
    assert result.opportunity_state == "opportunity_activated"
