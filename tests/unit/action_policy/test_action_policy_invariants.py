from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from behavior.core.model.behavior_pattern import BehaviorPattern
from behavior.projection import behavior_pattern_to_context_object
from policy.action_policy.decision.action_context import ActionContext
from policy.action_policy.decision.gate import PolicyGate
from policy.action_policy.model.action_policy import ActionCandidate, ActionPolicy, ActionPolicyStatus
from policy.action_policy.model.reward_signal import PenaltySignal, RewardSignal
from policy.action_policy.planning.session_commit_planner import ActionPolicyCommitPlanner
from policy.action_policy.ranking.action_policy_ranker import ActionPolicyRanker
from policy.action_policy.update.action_policy_factory import ActionPolicyEvidence, ActionPolicyFactory
from policy.action_policy.update.action_policy_updater import ActionPolicyUpdater
from pre.session import SessionArchive
from tests.support.persistence import FileSystemSourceStore, InMemoryIndexStore, seed_context_object
from transaction.model.operation_action import OperationAction


def _policy(**overrides) -> ActionPolicy:
    values = {
        "user_id": "u1",
        "scene_key": "hot",
        "action": "turn_on_fan",
        "support_anchor_uri": "memoryos://user/u1/support/behavior/hot",
        "auto_execute_allowed": True,
        "confidence": 0.95,
        "q_value": 0.95,
    }
    values.update(overrides)
    return ActionPolicy(**values)


def _context(policy: ActionPolicy) -> ActionContext:
    return ActionContext(
        user_id=policy.user_id,
        candidate_actions=[policy.action],
        packed_context={
            "slices": {
                "support_anchor": {
                    "items": [
                        {
                            "uri": policy.support_anchor_uri,
                            "context_type": "behavior_support",
                            "verified_exact_anchor": True,
                        }
                    ]
                }
            }
        },
    )


def test_confirmation_required_action_is_fail_closed_when_loaded() -> None:
    policy = _policy(action="turn_on_ac", auto_execute_allowed=True)

    assert policy.auto_execute_allowed is False


def test_candidate_must_match_action_policy_identity() -> None:
    policy = _policy()
    candidate = ActionCandidate(
        action=policy.action,
        score=0.95,
        policy_uri="memoryos://user/u1/action_policies/other/turn_on_fan",
        reason="test",
    )

    decision = PolicyGate().evaluate(candidate, _context(policy), policy, 0.95)

    assert decision.mode == "blocked"


def test_cooldown_blocks_until_deadline_and_then_resumes() -> None:
    policy = _policy(status=ActionPolicyStatus.COOLDOWN)
    candidate = ActionCandidate(policy.action, 0.95, policy.uri, "test")
    policy.cooldown_until = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    assert PolicyGate().evaluate(candidate, _context(policy), policy, 0.95).mode == "ask_user"

    policy.cooldown_until = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    assert PolicyGate().evaluate(candidate, _context(policy), policy, 0.95).mode == "execute"


def test_ranker_removes_retired_policy_and_penalizes_cross_scene_fallback() -> None:
    retired = _policy(status=ActionPolicyStatus.OBSOLETE)
    exact = _policy()
    fallback = _policy(scene_key="warm", cross_scene_fallback=True)

    ranked = ActionPolicyRanker().rank([retired, fallback, exact])

    assert [candidate.policy_uri for candidate in ranked] == [exact.uri, fallback.uri]
    assert ranked[0].features["scene_scope_match"] == 1.0
    assert ranked[1].features["scene_scope_match"] == 0.0


def test_factory_refresh_uses_cumulative_snapshot_without_double_counting() -> None:
    evidence = ActionPolicyEvidence(
        user_id="u1",
        scene_key="hot",
        action="turn_on_fan",
        support_anchor_uri="memoryos://user/u1/support/behavior/hot",
        opportunity_count=3,
        activation_count=3,
        explicit_authorized=True,
        supported_behavior_pattern_uris=["memoryos://user/u1/behavior_patterns/hot"],
    )
    factory = ActionPolicyFactory()
    first = factory.build(evidence)
    replayed = factory.build(evidence, existing=first)

    assert replayed.success_count == first.success_count == 0
    assert replayed.opportunity_count == first.opportunity_count == 3
    assert replayed.reward_score == first.reward_score


def test_factory_treats_behavior_frequency_as_activation_not_reward() -> None:
    factory = ActionPolicyFactory()
    frequent = factory.build(
        ActionPolicyEvidence(
            user_id="u1",
            scene_key="hot",
            action="turn_on_fan",
            support_anchor_uri="memoryos://user/u1/support/behavior/hot",
            opportunity_count=10,
            activation_count=8,
            supported_behavior_pattern_uris=["memoryos://user/u1/behavior/patterns/hot/p1"],
        )
    )
    rare = factory.build(
        ActionPolicyEvidence(
            user_id="u1",
            scene_key="hot",
            action="drink_water",
            support_anchor_uri="memoryos://user/u1/support/behavior/hot",
            opportunity_count=10,
            activation_count=2,
            supported_behavior_pattern_uris=["memoryos://user/u1/behavior/patterns/hot/p1"],
        )
    )

    assert frequent.q_value > rare.q_value
    assert frequent.reward_score == rare.reward_score == 0.0
    assert frequent.success_count == rare.success_count == 0


def test_feedback_signals_reject_zero_and_updater_sets_bounded_cooldown() -> None:
    with pytest.raises(ValueError):
        RewardSignal(0.0)
    with pytest.raises(ValueError):
        PenaltySignal(0.0)

    policy = _policy()
    ActionPolicyUpdater().penalize(policy, PenaltySignal(0.5))

    assert policy.status == ActionPolicyStatus.COOLDOWN
    assert policy.cooldown_until is not None
    assert datetime.fromisoformat(policy.cooldown_until) > datetime.now(timezone.utc)


def test_decision_output_does_not_form_policy_or_turn_zero_feedback_positive() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="s1",
        archive_uri="memoryos://user/u1/sessions/history/s1",
        observations=[{"scene_key": "hot"}],
        predictions=[
            {
                "observation": {"scene_key": "hot"},
                "decision": {"action": "turn_on_fan"},
                "candidates": [{"action": "turn_on_fan", "score": 0.99}],
            }
        ],
        feedback=[{"scene_key": "hot", "action": "turn_on_fan", "reward": 0.0}],
    )

    assert ActionPolicyCommitPlanner().plan(archive) == []


def test_action_policy_planner_does_not_create_missing_behavior_support(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    pattern = BehaviorPattern(
        user_id="u1",
        scene_key="hot",
        trigger_conditions={"scene_key": "hot"},
        support_anchor_uri="memoryos://user/u1/support/behavior/hot_anchor",
        case_refs=["case-1", "case-2", "case-3"],
        action_distribution=[{"action": "turn_on_fan", "count": 3}],
    )
    seed_context_object(
        source,
        index,
        behavior_pattern_to_context_object(pattern),
        content="stable behavior without its support anchor",
    )
    archive = SessionArchive(
        user_id="u1",
        session_id="s-missing-anchor",
        archive_uri="memoryos://user/u1/sessions/history/s-missing-anchor",
        observations=[{"scene_key": "hot"}],
    )

    operations = ActionPolicyCommitPlanner(index, source).plan(archive)

    assert operations == []
    with pytest.raises(FileNotFoundError):
        source.read_object(pattern.support_anchor_uri)


def test_explicit_negative_feedback_has_one_disable_path() -> None:
    archive = SessionArchive(
        user_id="u1",
        session_id="s2",
        archive_uri="memoryos://user/u1/sessions/history/s2",
        observations=[{"scene_key": "hot"}],
        feedback=[
            {
                "policy_uri": "memoryos://user/u1/action_policies/hot/turn_on_fan",
                "reward": -1.0,
                "explicit_rule": "以后不要自动打开风扇",
            }
        ],
    )

    operations = ActionPolicyCommitPlanner().plan(archive)

    assert [operation.action for operation in operations] == [
        OperationAction.PENALIZE,
        OperationAction.ADD,
    ]
    assert operations[0].payload["explicit_rule"] == "以后不要自动打开风扇"
