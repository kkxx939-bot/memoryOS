from __future__ import annotations

import sqlite3

import pytest

from memoryos.adapters.persistence.in_memory.relation_store import InMemoryRelationStore
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.store.sqlite_relation_store import SQLiteRelationStore


class _ProtectedDomain:
    @staticmethod
    def owns_uri(uri: str) -> bool:
        return uri.startswith("memoryos://user/u1/protected/")

    @staticmethod
    def owns_object(obj: ContextObject) -> bool:
        return _ProtectedDomain.owns_uri(obj.uri)


def test_relation_store_rejects_non_greenfield_layout(tmp_path) -> None:
    path = tmp_path / "relations.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE relations (
              source_uri TEXT NOT NULL,
              relation_type TEXT NOT NULL,
              target_uri TEXT NOT NULL,
              PRIMARY KEY (source_uri, relation_type, target_uri)
            )
            """
        )

    with pytest.raises(RuntimeError, match="reset the greenfield runtime"):
        SQLiteRelationStore(path)


def test_sqlite_relation_store_filters_tenant_and_user_but_allows_global_targets(tmp_path) -> None:
    store = SQLiteRelationStore(tmp_path / "relations.sqlite3")
    policy_a = "memoryos://user/u1/action_policies/hot/turn_on_ac.json"
    policy_b = "memoryos://user/u2/action_policies/hot/turn_on_ac.json"
    memory_a = "memoryos://user/u1/memories/anchors/hot"
    resource = "memoryos://resources/devices/ac"

    store.add_relation(
        ContextRelation(
            source_uri=policy_a,
            relation_type="anchored_by",
            target_uri=memory_a,
            metadata={"tenant_id": "default", "owner_user_id": "u1"},
        ),
        tenant_id="default",
    )
    store.add_relation(
        ContextRelation(
            source_uri=policy_b,
            relation_type="anchored_by",
            target_uri="memoryos://user/u2/memories/anchors/hot",
            metadata={"tenant_id": "default", "owner_user_id": "u2"},
        ),
        tenant_id="default",
    )
    store.add_relation(
        ContextRelation(
            source_uri=policy_a,
            relation_type="requires_resource",
            target_uri=resource,
            metadata={"tenant_id": "default", "owner_user_id": "u1"},
        ),
        tenant_id="default",
    )
    store.add_relation(
        ContextRelation(
            source_uri=policy_a,
            relation_type="requires_skill",
            target_uri="memoryos://skills/ac-control",
            metadata={"tenant_id": "other", "owner_user_id": "u1"},
        ),
        tenant_id="other",
    )

    user_a = store.relations_of(policy_a, tenant_id="default", owner_user_id="u1")
    assert {relation.target_uri for relation in user_a} == {memory_a, resource}

    user_b = store.relations_of(policy_a, tenant_id="default", owner_user_id="u2")
    assert user_b == []


def test_identical_relation_triples_coexist_and_delete_by_tenant(tmp_path) -> None:  # noqa: ANN001
    store = SQLiteRelationStore(tmp_path / "relations.sqlite3")
    source = "memoryos://user/shared/memory/documents/memdoc_same"
    target = "memoryos://user/shared/resources/same-target"
    for tenant_id in ("tenant-a", "tenant-b"):
        store.add_relation(
            ContextRelation(
                source_uri=source,
                relation_type="references",
                target_uri=target,
                metadata={"tenant_id": tenant_id, "owner_user_id": "shared"},
            ),
            tenant_id=tenant_id,
        )

    assert len(store.relations_of(source, tenant_id="tenant-a")) == 1
    assert len(store.relations_of(source, tenant_id="tenant-b")) == 1


def test_in_memory_relation_identity_and_reads_are_tenant_qualified() -> None:
    store = InMemoryRelationStore()
    source = "memoryos://user/shared/resources/source"
    target = "memoryos://resources/shared-target"
    for tenant_id in ("tenant-a", "tenant-b"):
        store.add_relation(
            ContextRelation(
                source_uri=source,
                relation_type="references",
                target_uri=target,
                metadata={"tenant_id": tenant_id},
            ),
            tenant_id=tenant_id,
        )

    assert len(store.all_relations(tenant_id="tenant-a")) == 1
    assert len(store.all_relations(tenant_id="tenant-b")) == 1
    store.delete_relation(
        source,
        "references",
        target,
        tenant_id="tenant-a",
    )
    assert store.relations_of(source, tenant_id="tenant-a") == []
    assert len(store.relations_of(source, tenant_id="tenant-b")) == 1
    with pytest.raises(TypeError):
        store.relations_of(source)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        store.delete_relation(source, "references", target)  # type: ignore[call-arg]

    store.delete_relation(source, "references", target, tenant_id="tenant-a")
    assert store.relations_of(source, tenant_id="tenant-a") == []
    assert len(store.relations_of(source, tenant_id="tenant-b")) == 1


def test_ordinary_reconcile_and_clear_are_tenant_and_domain_source_scoped(tmp_path) -> None:  # noqa: ANN001
    store = SQLiteRelationStore(
        tmp_path / "relations.sqlite3",
        domain_classifier=_ProtectedDomain(),
    )
    ordinary_source = "memoryos://user/u1/action_policies/hot/turn_on_ac"
    protected_target = "memoryos://user/u1/protected/documents/d1/blocks/b1"
    protected_source = "memoryos://user/u1/protected/documents/d1"
    ordinary_target = "memoryos://user/u1/memories/rules/r1"
    wanted = ContextRelation(
        source_uri=ordinary_source,
        relation_type="constrained_by",
        target_uri=protected_target,
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
        ),
        tenant_id="tenant-a",
    )
    protected_row = ContextRelation(
        source_uri=protected_source,
        relation_type="contains",
        target_uri=protected_target,
        metadata={"tenant_id": "tenant-a", "owner_user_id": "u1"},
    )
    store.add_relation(protected_row, tenant_id="tenant-a")
    other_tenant = ContextRelation(
        source_uri=ordinary_source,
        relation_type="stale",
        target_uri=ordinary_target,
        metadata={"tenant_id": "tenant-b", "owner_user_id": "u1"},
    )
    store.add_relation(other_tenant, tenant_id="tenant-b")

    assert store.clear_ordinary_relations(tenant_id="tenant-a", limit=1) == 1
    assert store.clear_ordinary_relations(tenant_id="tenant-a", limit=10) == 1
    assert store.clear_ordinary_relations(tenant_id="tenant-a", limit=10) == 0
    assert store.relations_of(protected_source, tenant_id="tenant-a") == [protected_row]
    assert store.relations_of(ordinary_source, tenant_id="tenant-b") == [other_tenant]

    restored = store.reconcile_ordinary_relations((wanted,), tenant_id="tenant-a")
    assert restored["written"] == 1
    with pytest.raises(ValueError, match="domain-owned Source"):
        store.reconcile_ordinary_relations((protected_row,), tenant_id="tenant-a")
    assert store.relations_of(protected_source, tenant_id="tenant-a") == [protected_row]
