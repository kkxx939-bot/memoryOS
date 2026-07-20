from __future__ import annotations

from behavior.core.model.behavior_pattern import BehaviorPattern
from behavior.core.support import BehaviorSupportAnchor
from behavior.projection import behavior_pattern_to_context_object, behavior_support_to_context_object
from infrastructure.context.layers.generator import (
    generate_l0_for_object,
    generate_l1_for_object,
    l0_abstract,
    l1_overview,
)
from policy.action_policy.model.action_policy import ActionPolicy
from policy.action_policy.model.policy_support_rule import PolicySupportRule
from policy.action_policy.update.policy_support_writer import policy_support_rule_to_context_object


def test_behavior_pattern_structured_l0_l1() -> None:
    pattern = behavior_pattern_to_context_object(
        BehaviorPattern(
            user_id="u1",
            scene_key="hot_room",
            trigger_conditions={"context_tags": ["home"]},
            support_anchor_uri="memoryos://user/u1/support/behavior/hot",
            case_refs=["c1", "c2", "c3"],
            action_distribution=[{"action": "turn_on_ac", "count": 3}],
        )
    )

    assert "hot_room" in generate_l0_for_object(pattern, "")
    l1 = generate_l1_for_object(pattern, "")
    assert "# BehaviorPattern: hot_room" in l1
    assert "Dominant Actions:" in l1


def test_action_policy_structured_l0_l1() -> None:
    policy = ActionPolicy(
        user_id="u1",
        scene_key="hot_room",
        action="turn_on_ac",
        support_anchor_uri="memoryos://user/u1/support/behavior/hot",
        q_value=0.8,
        confidence=0.7,
    ).to_context_object()

    assert "q_value=0.8" in generate_l0_for_object(policy, "")
    l1 = generate_l1_for_object(policy, "")
    assert "# ActionPolicy: hot_room/turn_on_ac" in l1
    assert "Relations:" in l1


def test_behavior_and_policy_support_structured_l0_l1() -> None:
    behavior_support = behavior_support_to_context_object(
        BehaviorSupportAnchor(
            uri="memoryos://user/u1/support/behavior/hot",
            user_id="u1",
            title="hot behavior support",
            content="recurring hot-room evidence",
            anchor_key="hot",
        )
    )
    policy_support = policy_support_rule_to_context_object(
        PolicySupportRule(
            uri="memoryos://user/u1/support/action-policy/no-auto",
            user_id="u1",
            title="no automatic AC",
            content="do not automatically turn on AC",
            rule_key="no-auto",
            policy_rule_type="action_auto_execute",
            policy_rule_value="forbidden",
        )
    )

    assert "行为支持锚点" in generate_l0_for_object(behavior_support, "")
    assert "规则值=forbidden" in generate_l0_for_object(policy_support, "")
    assert "# behavior_support" in generate_l1_for_object(behavior_support, "")
    assert "# action_policy_support" in generate_l1_for_object(policy_support, "")


def test_unknown_object_uses_safe_empty_fallback() -> None:
    assert generate_l0_for_object(object(), "fallback text")
    assert generate_l1_for_object(object(), "")


def test_existing_layer_functions_still_callable() -> None:
    assert l0_abstract("a " * 200, max_chars=10)
    assert l1_overview("Title", ["one"]).startswith("# Title")
