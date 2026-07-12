from __future__ import annotations

from memoryos.action_policy.model.action_policy import ActionCandidate, ActionPolicy
from memoryos.contextdb.model.context_layer import ContextLayers
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore, InMemoryRelationStore
from memoryos.prediction.pipeline.action_context_builder import ActionContextBuilder


def test_action_context_builder_prefers_l1_falls_back_to_l0_and_records_layers(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()
    anchor_uri = "memoryos://user/u1/memories/anchors/hot"
    anchor = ContextObject(
        uri=anchor_uri,
        context_type=ContextType.MEMORY,
        title="anchor",
        owner_user_id="u1",
        metadata={"memory_kind": "anchor_memory"},
        layers=ContextLayers(l0_uri=f"{anchor_uri}/.abstract.md", l1_uri=f"{anchor_uri}/.overview.md", l2_uri=f"{anchor_uri}/content.md"),
    )
    source.write_object(anchor, content="L2")
    assert anchor.layers.l0_uri and anchor.layers.l1_uri
    source.write_content(anchor.layers.l0_uri, "L0")
    source.write_content(anchor.layers.l1_uri, "L1")
    policy = ActionPolicy(user_id="u1", scene_key="hot", action="turn_on_ac", memory_anchor_uri=anchor_uri)
    relations.add_relation(ContextRelation(source_uri=policy.uri, relation_type="anchored_by", target_uri=anchor_uri, metadata={"owner_user_id": "u1"}))

    context = ActionContextBuilder(index, source_store=source, relation_store=relations).build(
        "u1",
        [ActionCandidate(action=policy.action, score=0.8, policy_uri=policy.uri, reason="test")],
        [policy],
        token_budget=1000,
    )
    item = context.packed_context["slices"]["memory_anchor"]["items"][0]
    assert item["content"] == "L1"
    assert item["layer"] == "l1"
    assert context.packed_context["load_plan"][0]["layer"] in {"l1", "metadata"}

    source.write_content(anchor.layers.l1_uri, "")
    source._content_path(anchor.layers.l1_uri).unlink()
    context = ActionContextBuilder(index, source_store=source, relation_store=relations).build(
        "u1",
        [ActionCandidate(action=policy.action, score=0.8, policy_uri=policy.uri, reason="test")],
        [policy],
        token_budget=1000,
    )
    item = context.packed_context["slices"]["memory_anchor"]["items"][0]
    assert item["content"] == "L0"
    assert item["layer"] == "l0"


def test_action_context_builder_loads_l2_only_for_strong_relevance_and_budget(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()
    anchor_uri = "memoryos://user/u1/memories/anchors/hot"
    anchor = ContextObject(
        uri=anchor_uri,
        context_type=ContextType.MEMORY,
        title="anchor",
        owner_user_id="u1",
        metadata={"memory_kind": "anchor_memory"},
        layers=ContextLayers(l0_uri=f"{anchor_uri}/.abstract.md", l1_uri=f"{anchor_uri}/.overview.md", l2_uri=f"{anchor_uri}/content.md"),
    )
    source.write_object(anchor, content="L2")
    assert anchor.layers.l0_uri and anchor.layers.l1_uri
    source.write_content(anchor.layers.l0_uri, "L0")
    source.write_content(anchor.layers.l1_uri, "L1")
    policy = ActionPolicy(user_id="u1", scene_key="hot", action="turn_on_ac", memory_anchor_uri=anchor_uri)
    relations.add_relation(ContextRelation(source_uri=policy.uri, relation_type="anchored_by", target_uri=anchor_uri, metadata={"owner_user_id": "u1"}))

    default_context = ActionContextBuilder(index, source_store=source, relation_store=relations).build(
        "u1",
        [ActionCandidate(action=policy.action, score=0.84, policy_uri=policy.uri, reason="test")],
        [policy],
        token_budget=2000,
    )
    assert default_context.packed_context["slices"]["memory_anchor"]["items"][0]["layer"] == "l1"

    strong_context = ActionContextBuilder(index, source_store=source, relation_store=relations).build(
        "u1",
        [ActionCandidate(action=policy.action, score=0.95, policy_uri=policy.uri, reason="test")],
        [policy],
        token_budget=2000,
    )
    assert strong_context.packed_context["slices"]["memory_anchor"]["items"][0]["layer"] == "l2"


def test_action_context_builder_reports_dropped_contexts_when_budget_is_tight(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()
    anchor_uri = "memoryos://user/u1/memories/anchors/hot"
    anchor = ContextObject(uri=anchor_uri, context_type=ContextType.MEMORY, title="anchor", owner_user_id="u1")
    source.write_object(anchor, content="anchor")
    policy = ActionPolicy(user_id="u1", scene_key="hot", action="turn_on_ac", memory_anchor_uri=anchor_uri)
    relations.add_relation(ContextRelation(source_uri=policy.uri, relation_type="anchored_by", target_uri=anchor_uri, metadata={"owner_user_id": "u1"}))

    context = ActionContextBuilder(index, source_store=source, relation_store=relations).build(
        "u1",
        [ActionCandidate(action=policy.action, score=0.9, policy_uri=policy.uri, reason="test")],
        [policy],
        token_budget=50,
    )
    assert context.packed_context["dropped_contexts"]
