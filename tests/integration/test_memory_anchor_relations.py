from __future__ import annotations

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.behavior.model.behavior_pattern import BehaviorCluster, BehaviorPattern
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore, InMemoryRelationStore
from memoryos.memory.model.memory import MemoryCandidate, MemoryKind
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


def test_anchor_and_candidate_relations_are_populated(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()
    committer = OperationCommitter(source, index, tmp_path, relation_store=relations)
    anchor_uri = "memoryos://user/u1/memories/anchors/hot"

    cluster = BehaviorCluster(user_id="u1", scene_key="hot", memory_anchor_uri=anchor_uri, case_refs=["case1", "case2"])
    cluster_uri = "memoryos://user/u1/behavior/clusters/hot/c1"
    committer.commit(
        "u1",
        [
            ContextOperation(
                user_id="u1",
                context_type=ContextType.BEHAVIOR_CLUSTER,
                action=OperationAction.ADD,
                target_uri=cluster_uri,
                payload={
                    "context_object": {
                        "uri": cluster_uri,
                        "context_type": "behavior_cluster",
                        "title": "cluster",
                        "owner_user_id": "u1",
                        "metadata": cluster.__dict__,
                    },
                    "content": "cluster",
                },
            )
        ],
    )
    assert any(
        relation.relation_type == "anchored_by" and relation.target_uri == anchor_uri
        for relation in relations.relations_of(cluster_uri, owner_user_id="u1")
    )

    pattern = BehaviorPattern(
        user_id="u1",
        scene_key="hot",
        trigger_conditions={},
        memory_anchor_uri=anchor_uri,
        case_refs=["case1"],
        action_distribution=[],
    )
    committer.commit(
        "u1",
        [
            ContextOperation(
                user_id="u1",
                context_type=ContextType.BEHAVIOR_PATTERN,
                action=OperationAction.ADD,
                target_uri=pattern.uri,
                payload={"context_object": pattern.to_context_object().to_dict(), "content": "pattern"},
            )
        ],
    )
    assert any(
        relation.relation_type == "anchored_by" for relation in relations.relations_of(pattern.uri, owner_user_id="u1")
    )

    policy = ActionPolicy(user_id="u1", scene_key="hot", action="turn_on_ac", memory_anchor_uri=anchor_uri)
    committer.commit(
        "u1",
        [
            ContextOperation(
                user_id="u1",
                context_type=ContextType.ACTION_POLICY,
                action=OperationAction.ADD,
                target_uri=policy.uri,
                payload={"context_object": policy.to_context_object().to_dict(), "content": "policy"},
            )
        ],
    )
    assert any(
        relation.relation_type == "anchored_by" for relation in relations.relations_of(policy.uri, owner_user_id="u1")
    )

    candidate = MemoryCandidate(
        uri="memoryos://user/u1/memories/candidates/temp",
        user_id="u1",
        title="temp candidate",
        content="candidate content",
        kind=MemoryKind.CANDIDATE,
        supporting_behavior_uris=[pattern.uri],
    )
    committer.commit(
        "u1",
        [
            ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.ADD,
                target_uri=candidate.uri,
                payload={"context_object": candidate.to_context_object().to_dict(), "content": candidate.content},
            )
        ],
    )
    assert any(
        relation.relation_type == "evidence_for" and relation.target_uri == pattern.uri
        for relation in relations.relations_of(candidate.uri, owner_user_id="u1")
    )
