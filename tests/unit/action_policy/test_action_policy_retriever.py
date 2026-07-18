from __future__ import annotations

import json

from memoryos.action_policy.model.action_policy import ActionPolicy, ActionPolicyStatus
from memoryos.action_policy.retrieval import ActionPolicyRetriever
from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore, InMemoryRelationStore


def _db(tmp_path) -> ContextDB:
    return ContextDB(FileSystemSourceStore(tmp_path), InMemoryIndexStore(), InMemoryRelationStore())


def _seed(db: ContextDB, policy: ActionPolicy, lifecycle: LifecycleState = LifecycleState.ACTIVE) -> None:
    obj = policy.to_context_object()
    obj.lifecycle_state = lifecycle
    db.seed_object(obj, content=json.dumps(policy.to_dict()))


def test_retriever_loads_filters_and_keeps_disabled_auto_execute(tmp_path) -> None:
    db = _db(tmp_path)
    active = ActionPolicy(user_id="u1", scene_key="hot", action="turn_on_ac", support_anchor_uri="memoryos://user/u1/support/behavior/hot")
    disabled = ActionPolicy(
        user_id="u1",
        scene_key="hot",
        action="turn_on_fan",
        support_anchor_uri="memoryos://user/u1/support/behavior/hot",
        status=ActionPolicyStatus.DISABLED_AUTO_EXECUTE,
    )
    other_user = ActionPolicy(user_id="u2", scene_key="hot", action="turn_on_ac", support_anchor_uri="memoryos://user/u2/support/behavior/hot")
    deleted = ActionPolicy(user_id="u1", scene_key="hot", action="drink_water", support_anchor_uri="memoryos://user/u1/support/behavior/hot")
    obsolete = ActionPolicy(user_id="u1", scene_key="hot", action="smoke", support_anchor_uri="memoryos://user/u1/support/behavior/hot")
    for policy in (active, disabled, other_user):
        _seed(db, policy)
    _seed(db, deleted, lifecycle=LifecycleState.DELETED)
    _seed(db, obsolete, lifecycle=LifecycleState.OBSOLETE)

    policies = ActionPolicyRetriever(db.index_store, db.source_store).retrieve(
        "u1",
        ["turn_on_ac", "turn_on_fan", "drink_water", "smoke"],
        scene_key="hot",
    )

    assert [policy.action for policy in policies] == ["turn_on_ac", "turn_on_fan"]
    assert policies[1].status == ActionPolicyStatus.DISABLED_AUTO_EXECUTE


def test_retriever_returns_empty_when_no_policy(tmp_path) -> None:
    db = _db(tmp_path)

    assert ActionPolicyRetriever(db.index_store, db.source_store).retrieve("u1", ["turn_on_ac"], scene_key="hot") == []
