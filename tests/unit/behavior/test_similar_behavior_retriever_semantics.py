from __future__ import annotations

from behavior.core.model.observation import Observation
from behavior.retrieval import SimilarBehaviorRetriever
from infrastructure.context.facade import ContextDB
from infrastructure.context.operation_effects import InfrastructureContextOperationEffects
from infrastructure.store.contracts.index import IndexHit
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_relation import ContextRelation
from infrastructure.store.model.context.context_type import ContextType
from infrastructure.store.model.context.lifecycle import LifecycleState
from tests.support.persistence import (
    FileSystemSourceStore,
    InMemoryIndexStore,
    InMemoryRelationStore,
    seed_context_object,
)
from tests.support.transaction import build_test_operation_committer as OperationCommitter


def _object(uri: str, context_type: ContextType, title: str, metadata: dict | None = None) -> ContextObject:
    return ContextObject(uri=uri, context_type=context_type, title=title, owner_user_id="u1", metadata=metadata or {})


def test_similar_behavior_retriever_final_semantics(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()
    db = ContextDB(
        source,
        index,
        relations,
        relation_committer=OperationCommitter(
            source,
            index,
            str(source.root),
            context_effects=InfrastructureContextOperationEffects(),
            relation_store=relations,
        ),
    )
    anchor = _object(
        "memoryos://user/u1/support/behavior/hot",
        ContextType.BEHAVIOR_SUPPORT,
        "hot support anchor",
        {"support_anchor_kind": "behavior"},
    )
    pattern = _object("memoryos://user/u1/behavior/patterns/hot/p1", ContextType.BEHAVIOR_PATTERN, "hot pattern", {"scene_key": "hot"})
    cluster = _object("memoryos://user/u1/behavior/clusters/hot/c1", ContextType.BEHAVIOR_CLUSTER, "hot cluster", {"scene_key": "hot"})
    policy = _object("memoryos://user/u1/action_policies/hot/turn_on_ac", ContextType.ACTION_POLICY, "policy")
    cases = [
        _object(f"memoryos://user/u1/behavior/cases/{idx}", ContextType.BEHAVIOR_CASE, f"hot case {idx}", {"reward": reward, "created_at": f"2026-01-0{idx}T00:00:00Z"})
        for idx, reward in enumerate([1.0, -1.0, 0.2, 0.0, 0.0], start=1)
    ]
    for obj in [anchor, pattern, cluster, policy, *cases]:
        seed_context_object(db.source_store, db.index_store, obj, content=f"hot room {obj.title}")
    for relation in [
        ContextRelation(source_uri=pattern.uri, relation_type="anchored_by", target_uri=anchor.uri, metadata={"owner_user_id": "u1"}),
        ContextRelation(source_uri=pattern.uri, relation_type="aggregated_from", target_uri=cases[0].uri, metadata={"owner_user_id": "u1"}),
        ContextRelation(source_uri=pattern.uri, relation_type="aggregated_from", target_uri=cases[1].uri, metadata={"owner_user_id": "u1"}),
        ContextRelation(source_uri=pattern.uri, relation_type="updates_policy", target_uri=policy.uri, metadata={"owner_user_id": "u1"}),
    ]:
        db.add_relation(relation)

    result = SimilarBehaviorRetriever(db.index_store, source_store=db.source_store, relation_store=db.relation_store).retrieve(
        "u1",
        Observation(user_id="u1", raw_text="hot room", location="home", environment={"temperature": 30}),
    )

    assert result["patterns"]
    assert result["clusters"]
    assert len(result["representative_cases"]) <= 3
    assert result["support_anchors"][0]["uri"] == anchor.uri
    assert result["policy_refs"][0]["uri"] == policy.uri
    assert policy.uri not in {item["uri"] for item in result["representative_cases"]}
    assert result["similarity_scores"][pattern.uri] > result["similarity_scores"][cases[0].uri]
    assert result["retrieval_trace"][anchor.uri]["source"] == "relation"
    assert result["retrieval_trace"][cases[0].uri]["relation_type"] == "aggregated_from"


def test_similar_behavior_retriever_excludes_inactive_support_relations(tmp_path) -> None:  # noqa: ANN001
    db = ContextDB(FileSystemSourceStore(tmp_path), InMemoryIndexStore(), InMemoryRelationStore())
    pattern = _object(
        "memoryos://user/u1/behavior/patterns/hot/pending-filter",
        ContextType.BEHAVIOR_PATTERN,
        "hot pending filter pattern",
        {"scene_key": "hot"},
    )
    seed_context_object(db.source_store, db.index_store, pattern, content="hot room")
    rejected_uris = []
    for lifecycle_state in (
        LifecycleState.PENDING,
        LifecycleState.RETRYABLE,
        LifecycleState.CONFIRMED,
        LifecycleState.ACTIVE,
    ):
        support = _object(
            f"memoryos://user/u1/support/behavior/{lifecycle_state.value}",
            ContextType.BEHAVIOR_SUPPORT,
            f"hot {lifecycle_state.value} support",
            {"support_anchor_kind": "wrong" if lifecycle_state == LifecycleState.ACTIVE else "behavior"},
        )
        support.lifecycle_state = lifecycle_state
        db.source_store.write_object(support, content="hot room invalid support")
        db.index_store.upsert_index(
            support,
            content="hot room invalid support",
            tenant_id="default",
        )
        db.relation_store.add_relation(
            ContextRelation(
                source_uri=pattern.uri,
                relation_type="anchored_by",
                target_uri=support.uri,
                metadata={"owner_user_id": "u1"},
            ),
            tenant_id="default",
        )
        rejected_uris.append(support.uri)

    result = SimilarBehaviorRetriever(
        db.index_store,
        source_store=db.source_store,
        relation_store=db.relation_store,
    ).retrieve(
        "u1",
        Observation(user_id="u1", raw_text="hot room", location="home"),
    )

    assert set(rejected_uris).isdisjoint(item["uri"] for item in result["support_anchors"])
    assert set(rejected_uris).isdisjoint(item["uri"] for item in result["hits"])

    source_less = SimilarBehaviorRetriever(
        db.index_store,
        relation_store=db.relation_store,
    ).retrieve(
        "u1",
        Observation(user_id="u1", raw_text="hot room", location="home"),
    )
    assert set(rejected_uris).isdisjoint(item["uri"] for item in source_less["support_anchors"])


def test_similar_behavior_retriever_rejects_stale_support_index_hit(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    pattern = _object(
        "memoryos://user/u1/behavior/patterns/hot/stale-index",
        ContextType.BEHAVIOR_PATTERN,
        "hot stale index pattern",
    )
    pending = _object(
        "memoryos://user/u1/support/behavior/stale-index",
        ContextType.BEHAVIOR_SUPPORT,
        "hot stale pending support",
        {"support_anchor_kind": "behavior"},
    )
    pending.lifecycle_state = LifecycleState.PENDING
    cross_user = ContextObject(
        uri="memoryos://user/u2/support/behavior/cross-user-stale-index",
        context_type=ContextType.BEHAVIOR_SUPPORT,
        title="hot cross-user support",
        owner_user_id="u2",
        metadata={"support_anchor_kind": "behavior"},
    )
    source.write_object(pattern, content="hot room")
    source.write_object(pending, content="hot pending support")
    source.write_object(cross_user, content="hot private support")
    relations = InMemoryRelationStore()
    relations.add_relation(
        ContextRelation(
            source_uri=pattern.uri,
            relation_type="anchored_by",
            target_uri=cross_user.uri,
            metadata={"owner_user_id": "u1"},
        ),
        tenant_id="default",
    )

    class StalePendingIndex(InMemoryIndexStore):
        def search(  # noqa: ANN201
            self,
            query,  # noqa: ANN001, ARG002
            *,
            tenant_id,  # noqa: ANN001, ARG002
            filters=None,  # noqa: ANN001
            limit=10,  # noqa: ARG002
        ):
            context_type = dict(filters or {}).get("context_type")
            if context_type == ContextType.BEHAVIOR_PATTERN.value:
                return [
                    IndexHit(
                        uri=pattern.uri,
                        score=1.0,
                        context_type=ContextType.BEHAVIOR_PATTERN.value,
                        title=pattern.title,
                    )
                ]
            if context_type == ContextType.BEHAVIOR_SUPPORT.value:
                return [
                    IndexHit(
                        uri=pending.uri,
                        score=1.0,
                        context_type=ContextType.BEHAVIOR_SUPPORT.value,
                        title=pending.title,
                    ),
                    IndexHit(
                        uri=cross_user.uri,
                        score=1.0,
                        context_type=ContextType.BEHAVIOR_SUPPORT.value,
                        title=cross_user.title,
                    ),
                ]
            return []

    result = SimilarBehaviorRetriever(
        StalePendingIndex(),
        source_store=source,
        relation_store=relations,
    ).retrieve(
        "u1",
        Observation(user_id="u1", raw_text="hot room", location="home"),
    )

    assert pending.uri not in {item["uri"] for item in result["support_anchors"]}
    assert pending.uri not in {item["uri"] for item in result["hits"]}
    assert cross_user.uri not in {item["uri"] for item in result["support_anchors"]}
    assert cross_user.uri not in {item["uri"] for item in result["hits"]}
