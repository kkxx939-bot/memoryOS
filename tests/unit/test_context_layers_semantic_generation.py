from __future__ import annotations

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.behavior.model.behavior_pattern import BehaviorPattern
from memoryos.contextdb.layers.layer_generator import (
    generate_l0_for_object,
    generate_l1_for_object,
    l0_abstract,
    l1_overview,
)
from memoryos.memory.model.memory import Memory, MemoryKind


def test_behavior_pattern_structured_l0_l1() -> None:
    pattern = BehaviorPattern(
        user_id="u1",
        scene_key="hot_room",
        trigger_conditions={"context_tags": ["home"]},
        memory_anchor_uri="memoryos://user/u1/memories/anchors/hot",
        case_refs=["c1", "c2", "c3"],
        action_distribution=[{"action": "turn_on_ac", "count": 3}],
    ).to_context_object()

    assert "hot_room" in generate_l0_for_object(pattern, "")
    l1 = generate_l1_for_object(pattern, "")
    assert "# BehaviorPattern: hot_room" in l1
    assert "Dominant Actions:" in l1


def test_action_policy_structured_l0_l1() -> None:
    policy = ActionPolicy(
        user_id="u1",
        scene_key="hot_room",
        action="turn_on_ac",
        memory_anchor_uri="memoryos://user/u1/memories/anchors/hot",
        q_value=0.8,
        confidence=0.7,
    ).to_context_object()

    assert "q_value=0.8" in generate_l0_for_object(policy, "")
    l1 = generate_l1_for_object(policy, "")
    assert "# ActionPolicy: hot_room/turn_on_ac" in l1
    assert "Relations:" in l1


def test_memory_structured_l0_l1_and_empty_fallback() -> None:
    memory = Memory(
        uri="memoryos://user/u1/memories/m1",
        user_id="u1",
        title="hot preference",
        content="User prefers AC in hot rooms.",
        kind=MemoryKind.EXPLICIT,
    ).to_context_object()

    assert "hot preference" in generate_l0_for_object(memory, "")
    assert "# Memory: hot preference" in generate_l1_for_object(memory, "")
    assert generate_l0_for_object(object(), "fallback text")
    assert generate_l1_for_object(object(), "")


def test_existing_layer_functions_still_callable() -> None:
    assert l0_abstract("a " * 200, max_chars=10)
    assert l1_overview("Title", ["one"]).startswith("# Title")
