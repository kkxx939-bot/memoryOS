from __future__ import annotations

from memoryos.behavior.model.observation import Observation
from memoryos.behavior.retrieval import SimilarBehaviorRetriever
from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore, InMemoryRelationStore
from memoryos.contextdb.store.source_store import IndexHit
from memoryos.operations.commit.operation_committer import OperationCommitter


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
        committer=OperationCommitter(
            source,
            index,
            str(source.root),
            relation_store=relations,
        ),
    )
    anchor = _object("memoryos://user/u1/memories/anchors/hot", ContextType.MEMORY, "hot anchor")
    pattern = _object("memoryos://user/u1/behavior/patterns/hot/p1", ContextType.BEHAVIOR_PATTERN, "hot pattern", {"scene_key": "hot"})
    cluster = _object("memoryos://user/u1/behavior/clusters/hot/c1", ContextType.BEHAVIOR_CLUSTER, "hot cluster", {"scene_key": "hot"})
    policy = _object("memoryos://user/u1/action_policies/hot/turn_on_ac", ContextType.ACTION_POLICY, "policy")
    cases = [
        _object(f"memoryos://user/u1/behavior/cases/{idx}", ContextType.BEHAVIOR_CASE, f"hot case {idx}", {"reward": reward, "created_at": f"2026-01-0{idx}T00:00:00Z"})
        for idx, reward in enumerate([1.0, -1.0, 0.2, 0.0, 0.0], start=1)
    ]
    for obj in [anchor, pattern, cluster, policy, *cases]:
        db.seed_object(obj, content=f"hot room {obj.title}")
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
    assert result["memory_anchors"][0]["uri"] == anchor.uri
    assert result["policy_refs"][0]["uri"] == policy.uri
    assert policy.uri not in {item["uri"] for item in result["representative_cases"]}
    assert result["similarity_scores"][pattern.uri] > result["similarity_scores"][cases[0].uri]
    assert result["retrieval_trace"][anchor.uri]["source"] == "relation"
    assert result["retrieval_trace"][cases[0].uri]["relation_type"] == "aggregated_from"


def test_similar_behavior_retriever_excludes_pending_memory_relations(tmp_path) -> None:  # noqa: ANN001
    db = ContextDB(FileSystemSourceStore(tmp_path), InMemoryIndexStore(), InMemoryRelationStore())
    pattern = _object(
        "memoryos://user/u1/behavior/patterns/hot/pending-filter",
        ContextType.BEHAVIOR_PATTERN,
        "hot pending filter pattern",
        {"scene_key": "hot"},
    )
    db.seed_object(pattern, content="hot room")
    pending_uris = []
    for lifecycle_state in (
        LifecycleState.PENDING,
        LifecycleState.RETRYABLE,
        LifecycleState.CONFIRMED,
        LifecycleState.ACTIVE,
    ):
        pending = _object(
            f"memoryos://user/u1/memories/pending/{lifecycle_state.value}",
            ContextType.MEMORY,
            f"hot {lifecycle_state.value} memory",
            {
                "canonical_kind": "pending_proposal",
                "admission": {"decision": "pending"},
            },
        )
        pending.lifecycle_state = lifecycle_state
        # Deliberately construct an uncommitted raw pending artifact through
        # the low-level stores.  The public ContextDB boundary correctly
        # rejects this bypass; the retriever must still fail closed if such a
        # crash/migration artifact is present underneath it.
        db.source_store.write_object(pending, content="hot room unconfirmed memory")
        db.index_store.upsert_index(pending, content="hot room unconfirmed memory")
        db.relation_store.add_relation(
            ContextRelation(
                source_uri=pattern.uri,
                relation_type="anchored_by",
                target_uri=pending.uri,
                metadata={"owner_user_id": "u1"},
            )
        )
        pending_uris.append(pending.uri)

    result = SimilarBehaviorRetriever(
        db.index_store,
        source_store=db.source_store,
        relation_store=db.relation_store,
    ).retrieve(
        "u1",
        Observation(user_id="u1", raw_text="hot room", location="home"),
    )

    assert set(pending_uris).isdisjoint(item["uri"] for item in result["memory_anchors"])
    assert set(pending_uris).isdisjoint(item["uri"] for item in result["hits"])

    source_less = SimilarBehaviorRetriever(
        db.index_store,
        relation_store=db.relation_store,
    ).retrieve(
        "u1",
        Observation(user_id="u1", raw_text="hot room", location="home"),
    )
    assert set(pending_uris).isdisjoint(item["uri"] for item in source_less["memory_anchors"])


def test_similar_behavior_retriever_rejects_stale_pending_index_hit(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    pattern = _object(
        "memoryos://user/u1/behavior/patterns/hot/stale-index",
        ContextType.BEHAVIOR_PATTERN,
        "hot stale index pattern",
    )
    pending = _object(
        "memoryos://user/u1/memories/pending/stale-index",
        ContextType.MEMORY,
        "hot stale pending memory",
        {
            "canonical_kind": "pending_proposal",
            "admission": {"decision": "pending"},
        },
    )
    pending.lifecycle_state = LifecycleState.PENDING
    cross_user = ContextObject(
        uri="memoryos://user/u2/memories/anchors/cross-user-stale-index",
        context_type=ContextType.MEMORY,
        title="hot cross-user memory",
        owner_user_id="u2",
    )
    source.write_object(pattern, content="hot room")
    source.write_object(pending, content="hot pending memory")
    source.write_object(cross_user, content="hot private memory")
    relations = InMemoryRelationStore()
    relations.add_relation(
        ContextRelation(
            source_uri=pattern.uri,
            relation_type="anchored_by",
            target_uri=cross_user.uri,
            metadata={"owner_user_id": "u1"},
        )
    )

    class StalePendingIndex(InMemoryIndexStore):
        def search(self, query, filters=None, limit=10):  # noqa: ANN001, ANN201, ARG002
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
            if context_type == ContextType.MEMORY.value:
                return [
                    IndexHit(
                        uri=pending.uri,
                        score=1.0,
                        context_type=ContextType.MEMORY.value,
                        title=pending.title,
                    ),
                    IndexHit(
                        uri=cross_user.uri,
                        score=1.0,
                        context_type=ContextType.MEMORY.value,
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

    assert pending.uri not in {item["uri"] for item in result["memory_anchors"]}
    assert pending.uri not in {item["uri"] for item in result["hits"]}
    assert cross_user.uri not in {item["uri"] for item in result["memory_anchors"]}
    assert cross_user.uri not in {item["uri"] for item in result["hits"]}
