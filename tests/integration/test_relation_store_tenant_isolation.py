from __future__ import annotations

import pytest

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


def test_identical_relation_triples_coexist_and_delete_by_tenant(tmp_path) -> None:  # noqa: ANN001
    store = SQLiteRelationStore(tmp_path / "relations.sqlite3")
    source = "memoryos://user/shared/memories/canonical/slots/same"
    target = f"{source}/claims/same"
    for tenant_id in ("tenant-a", "tenant-b"):
        store.add_relation(
            ContextRelation(
                source_uri=source,
                relation_type="has_claim",
                target_uri=target,
                metadata={"tenant_id": tenant_id, "owner_user_id": "shared"},
            )
        )

    assert len(store.relations_of(source, tenant_id="tenant-a")) == 1
    assert len(store.relations_of(source, tenant_id="tenant-b")) == 1
    with pytest.raises(ValueError, match="tenant_id is required"):
        store.delete_relation(source, "has_claim", target)

    store.delete_relation(source, "has_claim", target, tenant_id="tenant-a")
    assert store.relations_of(source, tenant_id="tenant-a") == []
    assert len(store.relations_of(source, tenant_id="tenant-b")) == 1


def test_ordinary_reconcile_and_clear_are_tenant_and_canonical_source_scoped(tmp_path) -> None:  # noqa: ANN001
    store = SQLiteRelationStore(tmp_path / "relations.sqlite3")
    ordinary_source = "memoryos://user/u1/action_policies/hot/turn_on_ac"
    canonical_target = "memoryos://user/u1/memories/canonical/slots/s1/claims/c1"
    canonical_source = "memoryos://user/u1/memories/canonical/slots/s1"
    ordinary_target = "memoryos://user/u1/memories/rules/r1"
    wanted = ContextRelation(
        source_uri=ordinary_source,
        relation_type="constrained_by",
        target_uri=canonical_target,
        metadata={"tenant_id": "tenant-a", "owner_user_id": "u1"},
        created_at="2026-07-16T00:00:00+00:00",
    )
    assert store.reconcile_ordinary_relations((wanted,), tenant_id="tenant-a") == {
        "processed": 1,
        "written": 1,
        "skipped": 0,
    }
    assert store.reconcile_ordinary_relations((wanted,), tenant_id="tenant-a") == {
        "processed": 1,
        "written": 0,
        "skipped": 1,
    }
    store.add_relation(
        ContextRelation(
            source_uri=ordinary_source,
            relation_type="stale",
            target_uri=ordinary_target,
            metadata={"tenant_id": "tenant-a", "owner_user_id": "u1"},
        )
    )
    canonical_receipt_row = ContextRelation(
        source_uri=canonical_source,
        relation_type="has_claim",
        target_uri=canonical_target,
        metadata={"tenant_id": "tenant-a", "owner_user_id": "u1"},
    )
    store.add_relation(canonical_receipt_row)
    other_tenant = ContextRelation(
        source_uri=ordinary_source,
        relation_type="stale",
        target_uri=ordinary_target,
        metadata={"tenant_id": "tenant-b", "owner_user_id": "u1"},
    )
    store.add_relation(other_tenant)

    assert store.clear_ordinary_relations(tenant_id="tenant-a", limit=1) == 1
    assert store.clear_ordinary_relations(tenant_id="tenant-a", limit=10) == 1
    assert store.clear_ordinary_relations(tenant_id="tenant-a", limit=10) == 0
    assert store.relations_of(canonical_source, tenant_id="tenant-a") == [canonical_receipt_row]
    assert store.relations_of(ordinary_source, tenant_id="tenant-b") == [other_tenant]

    restored = store.reconcile_ordinary_relations((wanted,), tenant_id="tenant-a")
    assert restored["written"] == 1
    with pytest.raises(ValueError, match="canonical Source"):
        store.reconcile_ordinary_relations((canonical_receipt_row,), tenant_id="tenant-a")
    assert store.relations_of(canonical_source, tenant_id="tenant-a") == [canonical_receipt_row]
