from __future__ import annotations

import json

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.action_policy.retrieval import ActionPolicyRetriever
from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore, InMemoryRelationStore


def _db(tmp_path) -> ContextDB:
    return ContextDB(FileSystemSourceStore(tmp_path), InMemoryIndexStore(), InMemoryRelationStore())


def _seed(db: ContextDB, policy: ActionPolicy) -> None:
    db.seed_object(policy.to_context_object(), content=json.dumps(policy.to_dict()))


def test_exact_scene_policies_do_not_mix_cross_scene_when_actions_are_covered(tmp_path) -> None:
    db = _db(tmp_path)
    exact = ActionPolicy(user_id="u1", scene_key="hot_home", action="turn_on_ac", memory_anchor_uri="memoryos://user/u1/memories/anchors/hot")
    cross = ActionPolicy(user_id="u1", scene_key="hot_office", action="turn_on_ac", memory_anchor_uri="memoryos://user/u1/memories/anchors/hot", q_value=0.99)
    _seed(db, exact)
    _seed(db, cross)

    policies = ActionPolicyRetriever(db.index_store, db.source_store).retrieve("u1", ["turn_on_ac"], scene_key="hot_home")

    assert [policy.uri for policy in policies] == [exact.uri]
    assert policies[0].cross_scene_fallback is False


def test_cross_scene_policy_only_fills_when_exact_scene_is_insufficient(tmp_path) -> None:
    db = _db(tmp_path)
    exact = ActionPolicy(user_id="u1", scene_key="hot_home", action="turn_on_ac", memory_anchor_uri="memoryos://user/u1/memories/anchors/hot")
    fallback = ActionPolicy(user_id="u1", scene_key="hot_office", action="turn_on_fan", memory_anchor_uri="memoryos://user/u1/memories/anchors/hot", q_value=0.99)
    _seed(db, exact)
    _seed(db, fallback)

    policies = ActionPolicyRetriever(db.index_store, db.source_store).retrieve("u1", ["turn_on_ac", "turn_on_fan"], scene_key="hot_home")

    assert [policy.action for policy in policies] == ["turn_on_ac", "turn_on_fan"]
    assert policies[0].cross_scene_fallback is False
    assert policies[1].cross_scene_fallback is True
