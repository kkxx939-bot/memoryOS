from __future__ import annotations

from behavior.core.model.behavior_pattern import BehaviorCluster, BehaviorPattern
from behavior.core.support import BehaviorSupportAnchor
from behavior.projection import behavior_pattern_to_context_object, behavior_support_to_context_object
from infrastructure.context.operation_effects import InfrastructureContextOperationEffects
from infrastructure.store.model.context.context_type import ContextType
from policy.action_policy.integration.commit_registration import build_action_policy_transaction_extensions
from policy.action_policy.model.action_policy import ActionPolicy
from tests.support.persistence import FileSystemSourceStore, InMemoryIndexStore, InMemoryRelationStore
from tests.support.transaction import build_test_operation_committer as OperationCommitter
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_action import OperationAction


def test_anchor_and_candidate_relations_are_populated(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()
    committer = OperationCommitter(
        source,
        index,
        tmp_path,
        relation_store=relations,
        context_effects=InfrastructureContextOperationEffects(),
        domain_extensions=build_action_policy_transaction_extensions(),
    )
    anchor_uri = "memoryos://user/u1/support/behavior/hot"

    cluster = BehaviorCluster(user_id="u1", scene_key="hot", support_anchor_uri=anchor_uri, case_refs=["case1", "case2"])
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
        for relation in relations.relations_of(
            cluster_uri,
            tenant_id="default",
            owner_user_id="u1",
        )
    )

    pattern = BehaviorPattern(
        user_id="u1",
        scene_key="hot",
        trigger_conditions={},
        support_anchor_uri=anchor_uri,
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
                payload={"context_object": behavior_pattern_to_context_object(pattern).to_dict(), "content": "pattern"},
            )
        ],
    )
    assert any(
        relation.relation_type == "anchored_by"
        for relation in relations.relations_of(
            pattern.uri,
            tenant_id="default",
            owner_user_id="u1",
        )
    )

    policy = ActionPolicy(user_id="u1", scene_key="hot", action="turn_on_ac", support_anchor_uri=anchor_uri)
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
        relation.relation_type == "anchored_by"
        for relation in relations.relations_of(
            policy.uri,
            tenant_id="default",
            owner_user_id="u1",
        )
    )

    candidate = BehaviorSupportAnchor(
        uri=anchor_uri,
        user_id="u1",
        title="behavior support",
        content="behavior support evidence",
        anchor_key="hot",
        supporting_behavior_uris=[pattern.uri],
    )
    committer.commit(
        "u1",
        [
            ContextOperation(
                user_id="u1",
                context_type=ContextType.BEHAVIOR_SUPPORT,
                action=OperationAction.ADD,
                target_uri=candidate.uri,
                payload={
                    "context_object": behavior_support_to_context_object(candidate).to_dict(),
                    "content": candidate.content,
                },
            )
        ],
    )
    assert any(
        relation.relation_type == "evidence_for" and relation.target_uri == pattern.uri
        for relation in relations.relations_of(
            candidate.uri,
            tenant_id="default",
            owner_user_id="u1",
        )
    )
