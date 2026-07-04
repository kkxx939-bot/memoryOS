from __future__ import annotations

from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.store.sqlite_relation_store import SQLiteRelationStore


def test_sqlite_relation_store_filters_tenant_and_user_but_allows_global_targets(tmp_path) -> None:
    store = SQLiteRelationStore(tmp_path / "relations.sqlite3")
    policy_a = "memoryos://user/u1/action_policies/hot/turn_on_ac.json"
    policy_b = "memoryos://user/u2/action_policies/hot/turn_on_ac.json"
    memory_a = "memoryos://user/u1/memories/anchors/hot"
    resource = "memoryos://resources/devices/ac"

    store.add_relation(ContextRelation(source_uri=policy_a, relation_type="anchored_by", target_uri=memory_a, metadata={"tenant_id": "default", "owner_user_id": "u1"}))
    store.add_relation(ContextRelation(source_uri=policy_b, relation_type="anchored_by", target_uri="memoryos://user/u2/memories/anchors/hot", metadata={"tenant_id": "default", "owner_user_id": "u2"}))
    store.add_relation(ContextRelation(source_uri=policy_a, relation_type="requires_resource", target_uri=resource, metadata={"tenant_id": "default", "owner_user_id": "u1"}))
    store.add_relation(ContextRelation(source_uri=policy_a, relation_type="requires_skill", target_uri="memoryos://skills/ac-control", metadata={"tenant_id": "other", "owner_user_id": "u1"}))

    user_a = store.relations_of(policy_a, tenant_id="default", owner_user_id="u1")
    assert {relation.target_uri for relation in user_a} == {memory_a, resource}

    user_b = store.relations_of(policy_a, tenant_id="default", owner_user_id="u2")
    assert user_b == []
