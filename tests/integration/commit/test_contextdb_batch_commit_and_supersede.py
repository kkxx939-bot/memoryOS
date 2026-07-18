from __future__ import annotations

import json

from memoryos.action_policy.model.action_policy import ActionCandidate, ActionPolicy
from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore, InMemoryRelationStore
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.prediction.pipeline.action_context_builder import ActionContextBuilder


def _context_db(tmp_path):
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()
    committer = OperationCommitter(source, index, str(tmp_path), relation_store=relations)
    return ContextDB(source, index, relations, committer=committer), source, index, relations


def test_commit_operations_batches_same_user_update_delete_for_coalescer(tmp_path) -> None:
    db, source, _, _ = _context_db(tmp_path)
    obj = ContextObject(
        uri="memoryos://user/u1/resources/profile/temp",
        context_type=ContextType.RESOURCE,
        title="temperature",
        owner_user_id="u1",
    )
    db.seed_object(obj, content="old")
    updated = ContextObject(
        uri=obj.uri,
        context_type=ContextType.RESOURCE,
        title="temperature updated",
        owner_user_id="u1",
    )

    results = db.commit_operations(
        [
            ContextOperation(
                user_id="u1",
                context_type=ContextType.RESOURCE,
                action=OperationAction.UPDATE,
                target_uri=obj.uri,
                payload={"context_object": updated.to_dict(), "content": "new"},
            ),
            ContextOperation(
                user_id="u1",
                context_type=ContextType.RESOURCE,
                action=OperationAction.DELETE,
                target_uri=obj.uri,
                payload={"reason": "remove"},
            ),
        ]
    )

    assert len(results) == 1
    assert [operation.action for operation in results[0].operations] == [OperationAction.DELETE]
    assert source.read_object(obj.uri).lifecycle_state == LifecycleState.DELETED


def test_commit_operations_batches_same_user_reward_penalty_for_conflict_resolver(tmp_path) -> None:
    db, source, _, _ = _context_db(tmp_path)
    policy = ActionPolicy(
        user_id="u1",
        scene_key="hot_room",
        action="turn_on_ac",
        support_anchor_uri="memoryos://user/u1/support/behavior/hot",
    )
    db.seed_object(policy.to_context_object(), content=json.dumps(policy.to_dict()))

    results = db.commit_operations(
        [
            ContextOperation(
                user_id="u1",
                context_type=ContextType.ACTION_POLICY,
                action=OperationAction.REWARD,
                target_uri=policy.uri,
                payload={"reward": 0.2, "signal_type": "implicit_positive"},
            ),
            ContextOperation(
                user_id="u1",
                context_type=ContextType.ACTION_POLICY,
                action=OperationAction.PENALIZE,
                target_uri=policy.uri,
                payload={"penalty": 0.7, "signal_type": "execution_failure"},
            ),
        ]
    )

    assert len(results) == 1
    assert [operation.action for operation in results[0].operations] == [OperationAction.PENALIZE]
    assert len(results[0].rejected_operations) == 1
    assert source.read_object(policy.uri).metadata["failure_count"] == 1


def test_supersede_marks_old_obsolete_and_action_context_uses_active_replacement(tmp_path) -> None:
    db, source, index, relations = _context_db(tmp_path)
    policy = ActionPolicy(
        user_id="u1",
        scene_key="hot_room",
        action="turn_on_ac",
        support_anchor_uri="memoryos://user/u1/support/behavior/hot",
    )
    db.seed_object(policy.to_context_object(), content=json.dumps(policy.to_dict()))
    old = ContextObject(
        uri="memoryos://user/u1/support/action-policy/old-ac",
        context_type=ContextType.ACTION_POLICY_SUPPORT,
        title="old ac rule",
        owner_user_id="u1",
        metadata={"support_anchor_kind": "action_policy", "constrains_policy_uris": [policy.uri]},
    )
    db.seed_object(old, content="old turn_on_ac rule")
    relations.add_relation(
        ContextRelation(
            source_uri=policy.uri,
            relation_type="constrained_by",
            target_uri=old.uri,
            metadata={"owner_user_id": "u1"},
        ),
        tenant_id="default",
    )
    new = ContextObject(
        uri="memoryos://user/u1/support/action-policy/new-ac",
        context_type=ContextType.ACTION_POLICY_SUPPORT,
        title="new ac rule",
        owner_user_id="u1",
        metadata={"support_anchor_kind": "action_policy", "constrains_policy_uris": [policy.uri]},
    )

    db.commit_operation(
        ContextOperation(
            user_id="u1",
            context_type=ContextType.ACTION_POLICY_SUPPORT,
            action=OperationAction.SUPERSEDE,
            target_uri=old.uri,
            payload={"context_object": new.to_dict(), "content": "new turn_on_ac rule", "reason": "newer user preference"},
        )
    )

    old_obj = source.read_object(old.uri)
    new_obj = source.read_object(new.uri)
    assert old_obj.lifecycle_state == LifecycleState.OBSOLETE
    assert old_obj.metadata["superseded_by"] == new.uri
    assert new_obj.lifecycle_state == LifecycleState.ACTIVE
    relation_types = {
        (relation.source_uri, relation.relation_type, relation.target_uri)
        for relation in relations.relations_of(old.uri, tenant_id="default")
    }
    assert (new.uri, "supersedes", old.uri) in relation_types
    assert (old.uri, "superseded_by", new.uri) in relation_types
    assert old.uri not in [
        hit.uri
        for hit in index.search(
            "turn_on_ac",
            tenant_id="default",
            filters={"owner_user_id": "u1", "context_type": "action_policy_support"},
        )
    ]

    context = ActionContextBuilder(index, source_store=source, relation_store=relations).build(
        "u1",
        [ActionCandidate(action=policy.action, score=1.0, policy_uri=policy.uri, reason="test")],
        [policy],
        token_budget=1000,
    )
    assert old.uri not in context.source_uris
    assert new.uri in context.source_uris
