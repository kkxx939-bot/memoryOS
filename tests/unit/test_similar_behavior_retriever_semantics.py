from __future__ import annotations

from memoryos.behavior.model.observation import Observation
from memoryos.behavior.retrieval import SimilarBehaviorRetriever
from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore, InMemoryRelationStore


def _object(uri: str, context_type: ContextType, title: str, metadata: dict | None = None) -> ContextObject:
    return ContextObject(uri=uri, context_type=context_type, title=title, owner_user_id="u1", metadata=metadata or {})


def test_similar_behavior_retriever_final_semantics(tmp_path) -> None:
    db = ContextDB(FileSystemSourceStore(tmp_path), InMemoryIndexStore(), InMemoryRelationStore())
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
