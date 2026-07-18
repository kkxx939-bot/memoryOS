from __future__ import annotations

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.behavior.model.behavior_pattern import BehaviorPattern
from memoryos.contextdb.ordinary_relations import ordinary_relation_specs_for_object
from memoryos.support import SupportAnchor, SupportAnchorKind


def _edges(obj) -> set[tuple[str, str, str]]:  # noqa: ANN001
    return {
        (str(spec["source_uri"]), str(spec["relation_type"]), str(spec["target_uri"]))
        for spec in ordinary_relation_specs_for_object(obj)
    }


def test_behavior_and_action_policy_project_support_relations() -> None:
    anchor_uri = "memoryos://user/u1/support/behavior/hot"
    rule_uri = "memoryos://user/u1/support/action-policy/no-auto"
    pattern = BehaviorPattern(
        user_id="u1",
        scene_key="hot",
        trigger_conditions={},
        support_anchor_uri=anchor_uri,
        case_refs=[],
        action_distribution=[],
    )
    policy = ActionPolicy(
        user_id="u1",
        scene_key="hot",
        action="turn_on_ac",
        support_anchor_uri=anchor_uri,
        supported_behavior_pattern_uris=[pattern.uri],
        constrained_by_support_uris=[rule_uri],
    )

    assert (pattern.uri, "anchored_by", anchor_uri) in _edges(pattern.to_context_object())
    policy_edges = _edges(policy.to_context_object())
    assert (policy.uri, "anchored_by", anchor_uri) in policy_edges
    assert (policy.uri, "supported_by", pattern.uri) in policy_edges
    assert (policy.uri, "constrained_by", rule_uri) in policy_edges


def test_support_objects_project_only_support_owned_edges() -> None:
    pattern_uri = "memoryos://user/u1/behavior/patterns/hot/p1"
    policy_uri = "memoryos://user/u1/action_policies/hot/turn_on_ac"
    behavior_support = SupportAnchor(
        uri="memoryos://user/u1/support/behavior/hot",
        user_id="u1",
        title="hot support",
        content="evidence",
        anchor_key="hot",
        supporting_behavior_uris=[pattern_uri],
    )
    policy_support = SupportAnchor(
        uri="memoryos://user/u1/support/action-policy/no-auto",
        user_id="u1",
        title="no auto",
        content="do not automatically execute",
        anchor_key="no-auto",
        kind=SupportAnchorKind.ACTION_POLICY,
        constrains_policy_uris=[policy_uri],
        policy_rule_type="action_auto_execute",
        policy_rule_value="forbidden",
        related_action="turn_on_ac",
    )

    assert _edges(behavior_support.to_context_object()) == {
        (behavior_support.uri, "evidence_for", pattern_uri)
    }
    assert _edges(policy_support.to_context_object()) == {
        (policy_uri, "constrained_by", policy_support.uri)
    }
