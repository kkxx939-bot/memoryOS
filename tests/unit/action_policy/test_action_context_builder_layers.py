from __future__ import annotations

from infrastructure.store.model.context.context_layer import ContextLayers
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_relation import ContextRelation
from infrastructure.store.model.context.context_type import ContextType
from policy.action_policy.decision.context_builder import ActionContextBuilder
from policy.action_policy.model.action_policy import ActionCandidate, ActionPolicy
from tests.support.persistence import FileSystemSourceStore, InMemoryIndexStore, InMemoryRelationStore


def test_action_context_builder_prefers_l1_falls_back_to_l0_and_records_layers(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()
    anchor_uri = "memoryos://user/u1/support/behavior/hot"
    anchor = ContextObject(
        uri=anchor_uri,
        context_type=ContextType.BEHAVIOR_SUPPORT,
        title="anchor",
        owner_user_id="u1",
        metadata={"support_anchor_kind": "behavior"},
        layers=ContextLayers(l0_uri=f"{anchor_uri}/.abstract.md", l1_uri=f"{anchor_uri}/.overview.md", l2_uri=f"{anchor_uri}/content.md"),
    )
    source.write_object(anchor, content="L2")
    assert anchor.layers.l0_uri and anchor.layers.l1_uri
    source.write_content(anchor.layers.l0_uri, "L0")
    source.write_content(anchor.layers.l1_uri, "L1")
    policy = ActionPolicy(user_id="u1", scene_key="hot", action="turn_on_ac", support_anchor_uri=anchor_uri)
    relations.add_relation(
        ContextRelation(
            source_uri=policy.uri,
            relation_type="anchored_by",
            target_uri=anchor_uri,
            metadata={"owner_user_id": "u1"},
        ),
        tenant_id="default",
    )

    context = ActionContextBuilder(index, source_store=source, relation_store=relations).build(
        "u1",
        [ActionCandidate(action=policy.action, score=0.8, policy_uri=policy.uri, reason="test")],
        [policy],
    )
    item = context.packed_context["slices"]["support_anchor"]["items"][0]
    assert item["content"] == "L1"
    assert item["layer"] == "l1"
    assert context.packed_context["load_plan"][0]["layer"] in {"l1", "metadata"}

    source.write_content(anchor.layers.l1_uri, "")
    source._content_path(anchor.layers.l1_uri).unlink()
    context = ActionContextBuilder(index, source_store=source, relation_store=relations).build(
        "u1",
        [ActionCandidate(action=policy.action, score=0.8, policy_uri=policy.uri, reason="test")],
        [policy],
    )
    item = context.packed_context["slices"]["support_anchor"]["items"][0]
    assert item["content"] == "L0"
    assert item["layer"] == "l0"


def test_action_context_builder_loads_l2_only_for_strong_relevance(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()
    anchor_uri = "memoryos://user/u1/support/behavior/hot"
    anchor = ContextObject(
        uri=anchor_uri,
        context_type=ContextType.BEHAVIOR_SUPPORT,
        title="anchor",
        owner_user_id="u1",
        metadata={"support_anchor_kind": "behavior"},
        layers=ContextLayers(l0_uri=f"{anchor_uri}/.abstract.md", l1_uri=f"{anchor_uri}/.overview.md", l2_uri=f"{anchor_uri}/content.md"),
    )
    source.write_object(anchor, content="L2")
    assert anchor.layers.l0_uri and anchor.layers.l1_uri
    source.write_content(anchor.layers.l0_uri, "L0")
    source.write_content(anchor.layers.l1_uri, "L1")
    policy = ActionPolicy(user_id="u1", scene_key="hot", action="turn_on_ac", support_anchor_uri=anchor_uri)
    relations.add_relation(
        ContextRelation(
            source_uri=policy.uri,
            relation_type="anchored_by",
            target_uri=anchor_uri,
            metadata={"owner_user_id": "u1"},
        ),
        tenant_id="default",
    )

    default_context = ActionContextBuilder(index, source_store=source, relation_store=relations).build(
        "u1",
        [ActionCandidate(action=policy.action, score=0.84, policy_uri=policy.uri, reason="test")],
        [policy],
    )
    assert default_context.packed_context["slices"]["support_anchor"]["items"][0]["layer"] == "l1"

    strong_context = ActionContextBuilder(index, source_store=source, relation_store=relations).build(
        "u1",
        [ActionCandidate(action=policy.action, score=0.95, policy_uri=policy.uri, reason="test")],
        [policy],
    )
    assert strong_context.packed_context["slices"]["support_anchor"]["items"][0]["layer"] == "l2"


def test_action_context_builder_reports_items_over_section_limit(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()
    anchor_uri = "memoryos://user/u1/support/behavior/hot"
    anchor = ContextObject(
        uri=anchor_uri,
        context_type=ContextType.BEHAVIOR_SUPPORT,
        title="anchor",
        owner_user_id="u1",
        metadata={"support_anchor_kind": "behavior"},
    )
    source.write_object(anchor, content="anchor")
    policy = ActionPolicy(user_id="u1", scene_key="hot", action="turn_on_ac", support_anchor_uri=anchor_uri)
    relations.add_relation(
        ContextRelation(
            source_uri=policy.uri,
            relation_type="anchored_by",
            target_uri=anchor_uri,
            metadata={"owner_user_id": "u1"},
        ),
        tenant_id="default",
    )

    context = ActionContextBuilder(index, source_store=source, relation_store=relations).build(
        "u1",
        [ActionCandidate(action=policy.action, score=0.9, policy_uri=policy.uri, reason="test")],
        [policy],
        resources=[{"uri": f"memoryos://resources/{index}", "content": str(index)} for index in range(5)],
    )
    assert context.packed_context["dropped_contexts"]
    assert context.packed_context["dropped_contexts"][0]["reason"] == "section_limit"
