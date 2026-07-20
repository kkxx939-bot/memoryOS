from __future__ import annotations

import sqlite3
from dataclasses import replace
from typing import cast

import pytest

from infrastructure.context.maintenance.lifecycle_service import ContextLifecycleService
from infrastructure.context.maintenance.tombstone import ProjectionTombstoneService
from infrastructure.context.session_projector import SessionContextProjector
from infrastructure.store.contracts.vector import vector_row_id
from infrastructure.store.model.catalog import CatalogRecord, CatalogRecordKind
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_relation import ContextRelation
from infrastructure.store.model.context.context_type import ContextType
from infrastructure.store.operation.redo import RedoIntegrityError
from infrastructure.store.sqlite.index_store import SQLiteIndexStore
from infrastructure.store.sqlite.relation_store import SQLiteRelationStore
from openApi.sdk.client import MemoryOSClient
from pre.session import SessionArchive
from tests.support.persistence import FileSystemSourceStore, InMemoryVectorStore, seed_context_object
from tests.support.session_archive import build_session_archive_store
from tests.support.transaction import build_test_operation_committer as OperationCommitter
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_action import OperationAction


def _seed_client_context(
    client: MemoryOSClient,
    obj: ContextObject,
    *,
    content: str | bytes = "",
) -> None:
    seed_context_object(client.runtime.stores.source, client.runtime.stores.index, obj, content=content)


def _commit_client_operation(client: MemoryOSClient, operation: ContextOperation):  # noqa: ANN201
    return client.runtime.transaction.committer.commit(operation.user_id, [operation])


def test_tombstone_fails_closed_then_replays_all_derived_cleanup(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path, tenant_id="tenant-a")
    index = SQLiteIndexStore(tmp_path / "indexes" / "catalog.sqlite3")
    relations = SQLiteRelationStore(tmp_path / "indexes" / "relations.sqlite3")
    vectors = InMemoryVectorStore()
    uri = "memoryos://user/u1/resources/report"
    obj = ContextObject(
        uri=uri,
        context_type=ContextType.RESOURCE,
        title="quarterly report",
        tenant_id="tenant-a",
        owner_user_id="u1",
        metadata={
            "tree_paths": ["resources/desktop", "timeline/2026/07/14"],
            "source_kind": "resource",
        },
    )
    source.write_object(obj, content="quarterly revenue")
    index.upsert_index(obj, content="quarterly revenue", tenant_id="tenant-a")
    record = index.get_catalog_by_uri(uri, tenant_id="tenant-a")[0]
    row_id = vector_row_id("tenant-a", record.record_key)
    vectors.upsert_vector(
        row_id,
        [1.0, 0.0],
        metadata={
            "tenant_id": "tenant-a",
            "catalog_record_key": record.record_key,
            "source_revision": record.source_revision,
            "projection_effect_hash": record.projection_effect_hash,
        },
    )
    relations.add_relation(
        ContextRelation(
            source_uri=uri,
            relation_type="references",
            target_uri="memoryos://resources/shared",
            metadata={
                "tenant_id": "tenant-a",
                "owner_user_id": "u1",
                "catalog_record_key": record.record_key,
            },
        ),
        tenant_id="tenant-a",
    )
    service = ProjectionTombstoneService(
        index,
        source_store=source,
        vector_store=vectors,
        relation_store=relations,
    )

    tombstones = service.enqueue_uri(
        uri,
        tenant_id="tenant-a",
        reason="resource_deleted",
    )
    blocked = service.process_pending(tenant_id="tenant-a")
    assert blocked.failed == tombstones
    assert index.get_catalog(record.record_key, tenant_id="tenant-a") is not None
    assert row_id in vectors.vector_uris()
    assert relations.relations_of(uri, tenant_id="tenant-a")

    source.soft_delete(uri, "resource_deleted")
    replayed = service.process_pending(tenant_id="tenant-a")
    assert replayed.processed == tombstones
    assert index.get_catalog(record.record_key, tenant_id="tenant-a") is None
    assert index.search_catalog(
        "quarterly",
        tenant_id="tenant-a",
        filters={"tenant_id": "tenant-a"},
    ) == []
    assert row_id not in vectors.vector_uris()
    assert relations.relations_of(uri, tenant_id="tenant-a") == []
    assert source.read_object(uri).lifecycle_state.value == "deleted"


def test_context_lifecycle_service_owns_fact_and_projection_deletion(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path, tenant_id="tenant-a")
    index = SQLiteIndexStore(tmp_path / "indexes" / "catalog.sqlite3")
    relations = SQLiteRelationStore(tmp_path / "indexes" / "relations.sqlite3")
    obj = ContextObject(
        uri="memoryos://user/u1/resources/lifecycle-owned-delete",
        context_type=ContextType.RESOURCE,
        title="lifecycle owned delete",
        tenant_id="tenant-a",
        owner_user_id="u1",
    )
    seed_context_object(source, index, obj, content="lifecycle marker")
    tombstones = ProjectionTombstoneService(index, source_store=source, relation_store=relations)
    lifecycle = ContextLifecycleService(source, tombstones, tenant_id="tenant-a")

    result = lifecycle.delete_context(obj.uri)

    assert result["uri"] == obj.uri
    assert result["processed"]
    assert source.read_object(obj.uri).lifecycle_state.value == "deleted"
    assert index.get_catalog_by_uri(obj.uri, tenant_id="tenant-a") == []


def test_orphan_tombstone_deletes_hashed_vector_without_catalog_scan(tmp_path) -> None:  # noqa: ANN001
    index = SQLiteIndexStore(tmp_path / "indexes" / "catalog.sqlite3")
    vectors = InMemoryVectorStore()
    tenant_id = "tenant-a"
    source_uri = "memoryos://user/u1/sessions/history/orphan"
    original_record_key = "session:orphan:manifest:abc:root"
    row_id = vector_row_id(tenant_id, original_record_key)
    vectors.upsert_vector(
        row_id,
        [1.0, 0.0],
        metadata={
            "tenant_id": tenant_id,
            "catalog_record_key": original_record_key,
            "public_uri": f"{source_uri}/context/root",
            "source_uri": source_uri,
        },
    )
    service = ProjectionTombstoneService(index, vector_store=vectors)

    tombstones = service.enqueue_source_uri(
        source_uri,
        tenant_id=tenant_id,
        reason="orphan-session-cleanup",
    )
    result = service.process_tombstones(tombstones, tenant_id=tenant_id)

    assert result.processed == tombstones
    assert vectors.get_vector_metadata(row_id) is None
    assert vectors.vector_uris() == []


def test_orphan_tombstone_does_not_cross_tenant_vector_metadata(tmp_path) -> None:  # noqa: ANN001
    index = SQLiteIndexStore(tmp_path / "indexes" / "catalog.sqlite3")
    vectors = InMemoryVectorStore()
    source_uri = "memoryos://user/u1/sessions/history/shared"
    tenant_a_id = vector_row_id("tenant-a", "session:shared:a")
    tenant_b_id = vector_row_id("tenant-b", "session:shared:b")
    for tenant_id, row_id, record_key in (
        ("tenant-a", tenant_a_id, "session:shared:a"),
        ("tenant-b", tenant_b_id, "session:shared:b"),
    ):
        vectors.upsert_vector(
            row_id,
            [1.0, 0.0],
            metadata={
                "tenant_id": tenant_id,
                "catalog_record_key": record_key,
                "source_uri": source_uri,
            },
        )
    service = ProjectionTombstoneService(index, vector_store=vectors)

    tombstones = service.enqueue_source_uri(
        source_uri,
        tenant_id="tenant-a",
        reason="tenant-a-orphan-cleanup",
    )
    result = service.process_tombstones(tombstones, tenant_id="tenant-a")

    assert result.processed == tombstones
    assert vectors.get_vector_metadata(tenant_a_id) is None
    assert vectors.get_vector_metadata(tenant_b_id) is not None


def test_orphan_tombstone_deletes_owned_relations_without_crossing_tenant(tmp_path) -> None:  # noqa: ANN001
    index = SQLiteIndexStore(tmp_path / "indexes" / "catalog.sqlite3")
    relations = SQLiteRelationStore(tmp_path / "indexes" / "relations.sqlite3")
    source_uri = "memoryos://user/u1/sessions/history/orphan-relations"
    tenant_a_relation = ContextRelation(
        source_uri=source_uri,
        relation_type="used_context",
        target_uri="memoryos://resources/tenant-a-target",
        metadata={
            "tenant_id": "tenant-a",
            "catalog_record_key": "session:orphan:manifest:abc:root",
        },
    )
    tenant_b_relation = ContextRelation(
        source_uri=source_uri,
        relation_type="used_skill",
        target_uri="memoryos://skills/tenant-b-target",
        metadata={
            "tenant_id": "tenant-b",
            "catalog_record_key": "session:orphan:manifest:def:root",
        },
    )
    relations.add_relation(tenant_a_relation, tenant_id="tenant-a")
    relations.add_relation(tenant_b_relation, tenant_id="tenant-b")
    service = ProjectionTombstoneService(index, relation_store=relations)

    tombstones = service.enqueue_source_uri(
        source_uri,
        tenant_id="tenant-a",
        reason="orphan-relation-cleanup",
    )
    result = service.process_tombstones(tombstones, tenant_id="tenant-a")

    assert result.processed == tombstones
    assert relations.relations_of(source_uri, tenant_id="tenant-a") == []
    assert relations.relations_of(source_uri, tenant_id="tenant-b") == [tenant_b_relation]


def test_session_tombstone_removes_catalog_but_preserves_archive_evidence(tmp_path) -> None:  # noqa: ANN001
    archive_store = build_session_archive_store(tmp_path, tenant_id="tenant-a")
    archive = SessionArchive(
        user_id="u1",
        session_id="s-delete",
        archive_uri="memoryos://user/u1/sessions/history/s-delete",
        created_at="2026-07-14T03:00:00+00:00",
        metadata={"tenant_id": "tenant-a", "timezone": "UTC"},
        messages=[{"role": "user", "content": "keep immutable evidence"}],
    )
    archive_store.write_sync_archive(archive)
    index = SQLiteIndexStore(tmp_path / "indexes" / "catalog.sqlite3")
    SessionContextProjector(index).project(archive)
    source = FileSystemSourceStore(tmp_path, tenant_id="tenant-a")
    relations = SQLiteRelationStore(tmp_path / "indexes" / "relations.sqlite3")
    service = ProjectionTombstoneService(index, source_store=source, relation_store=relations)
    lifecycle = ContextLifecycleService(source, service, tenant_id="tenant-a")

    result = lifecycle.delete_session_context("s-delete")

    assert result["evidence_retained"] is True
    assert index.list_catalog(
        tenant_id="tenant-a",
        filters={"tenant_id": "tenant-a", "session_ids": ("s-delete",)},
    ) == []
    restored = archive_store.read_archive(archive.archive_uri, tenant_id="tenant-a")
    assert restored.messages[0]["content"] == "keep immutable evidence"


def test_session_tombstone_keyset_pages_past_one_thousand_records(tmp_path) -> None:  # noqa: ANN001
    index = SQLiteIndexStore(tmp_path / "indexes" / "catalog.sqlite3")
    timestamp = "2026-07-14T03:00:00+00:00"
    records = tuple(
        CatalogRecord(
            record_key=f"session:s-large:event:{ordinal:04d}",
            uri=f"memoryos://user/u1/sessions/history/s-large/context/event/{ordinal}",
            tenant_id="tenant-a",
            owner_user_id="u1",
            session_id="s-large",
            context_type="session",
            source_kind="tool_result",
            record_kind=CatalogRecordKind.TOOL_RESULT.value,
            tree_paths=("sessions/s-large", "timeline/2026/07/14"),
            created_at=timestamp,
            updated_at=timestamp,
            event_time=timestamp,
            ingested_at=timestamp,
            transaction_time=timestamp,
            title=f"tool result {ordinal}",
            l1_text=f"bounded result {ordinal}",
            source_uri="memoryos://user/u1/sessions/history/s-large",
            source_digest=f"digest-{ordinal}",
        )
        for ordinal in range(1_005)
    )
    assert index.upsert_catalog_batch(records, tenant_id="tenant-a") == 1_005
    source = FileSystemSourceStore(tmp_path, tenant_id="tenant-a")
    relations = SQLiteRelationStore(tmp_path / "indexes" / "relations.sqlite3")
    service = ProjectionTombstoneService(index, source_store=source, relation_store=relations)
    lifecycle = ContextLifecycleService(source, service, tenant_id="tenant-a")

    result = lifecycle.delete_session_context("s-large")

    # One durable Session barrier prevents a future projector version from
    # resurrecting newly introduced record kinds from immutable Archive data.
    assert len(result["tombstone_ids"]) == 1_006
    assert len(result["processed"]) == 1_006
    assert (
        index.scan_catalog_batch(
            tenant_id="tenant-a",
            filters={"tenant_id": "tenant-a", "session_ids": ("s-large"), "include_inactive": True},
            limit=1_000,
        )
        == []
    )
    with sqlite3.connect(index.path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM context_tombstones WHERE tenant_id = ? AND status = 'APPLIED'",
                ("tenant-a",),
            ).fetchone()[0]
            == 1_006
        )


def test_ordinary_delete_uses_durable_tombstone_and_drains_all_relations(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path), vector_store=InMemoryVectorStore())
    uri = "memoryos://user/u1/resources/large-report"
    obj = ContextObject(
        uri=uri,
        context_type=ContextType.RESOURCE,
        title="large report",
        tenant_id="default",
        owner_user_id="u1",
        metadata={"tree_paths": ["resources/desktop"], "source_kind": "resource"},
    )
    _seed_client_context(client, obj, content="durable delete marker")
    vectors = cast(InMemoryVectorStore, client.runtime.stores.vector)
    index = cast(SQLiteIndexStore, client.runtime.stores.index)
    record = index.get_catalog_by_uri(uri, tenant_id="default")[0]
    row_id = vector_row_id("default", record.record_key)
    vectors.upsert_vector(
        row_id,
        [1.0, 0.0],
        metadata={
            "tenant_id": "default",
            "catalog_record_key": record.record_key,
            "source_revision": record.source_revision,
            "projection_effect_hash": record.projection_effect_hash,
        },
    )
    for ordinal in range(1_005):
        client.runtime.stores.relation.add_relation(
            ContextRelation(
                source_uri=uri,
                relation_type="references",
                target_uri=f"memoryos://resources/shared-{ordinal}",
                metadata={
                    "tenant_id": "default",
                    "owner_user_id": "u1",
                    "catalog_record_key": record.record_key,
                },
            ),
            tenant_id="default",
        )

    operation = ContextOperation(
            operation_id="op_public_ordinary_delete",
            user_id="u1",
            context_type=ContextType.RESOURCE,
            action=OperationAction.DELETE,
            target_uri=uri,
            payload={"reason": "ordinary_delete"},
    )
    result = _commit_client_operation(client, operation)

    assert result.operations[0].status.value == "committed"
    assert result.operations[0].payload["projection_tombstone_ids"]
    assert client.runtime.stores.source.read_object(uri).lifecycle_state.value == "deleted"
    assert index.get_catalog_by_uri(uri, tenant_id="default") == []
    assert index.search_catalog(
        "durable delete marker",
        tenant_id="default",
        filters={"tenant_id": "default"},
    ) == []
    assert vectors.get_vector_metadata(row_id) is None
    assert client.runtime.stores.relation.relations_of(uri, tenant_id="default") == []
    with sqlite3.connect(index.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM context_paths WHERE uri = ?", (uri,)).fetchone()[0] == 0
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM context_tombstones WHERE uri = ? AND status = 'APPLIED'",
                (uri,),
            ).fetchone()[0]
            >= 1
        )


def test_contextdb_batch_delete_tombstones_every_serving_layer(tmp_path) -> None:  # noqa: ANN001
    vectors = InMemoryVectorStore()
    client = MemoryOSClient(str(tmp_path), vector_store=vectors)
    index = cast(SQLiteIndexStore, client.runtime.stores.index)
    operations: list[ContextOperation] = []
    vector_ids: list[str] = []
    uris = (
        "memoryos://user/u1/resources/batch-report-a",
        "memoryos://user/u1/resources/batch-report-b",
    )
    for ordinal, uri in enumerate(uris):
        obj = ContextObject(
            uri=uri,
            context_type=ContextType.RESOURCE,
            title=f"batch report {ordinal}",
            tenant_id="default",
            owner_user_id="u1",
            metadata={"tree_paths": ["resources/desktop"], "source_kind": "resource"},
        )
        _seed_client_context(client, obj, content=f"batch tombstone marker {ordinal}")
        record = index.get_catalog_by_uri(uri, tenant_id="default")[0]
        row_id = vector_row_id("default", record.record_key)
        vector_ids.append(row_id)
        vectors.upsert_vector(
            row_id,
            [1.0, float(ordinal)],
            metadata={
                "tenant_id": "default",
                "catalog_record_key": record.record_key,
                "source_revision": record.source_revision,
                "projection_effect_hash": record.projection_effect_hash,
            },
        )
        client.runtime.stores.relation.add_relation(
            ContextRelation(
                source_uri=uri,
                relation_type="references",
                target_uri=f"memoryos://resources/batch-target-{ordinal}",
                metadata={
                    "tenant_id": "default",
                    "owner_user_id": "u1",
                    "catalog_record_key": record.record_key,
                },
            ),
            tenant_id="default",
        )
        operations.append(
            ContextOperation(
                operation_id=f"op_batch_delete_{ordinal}",
                user_id="u1",
                context_type=ContextType.RESOURCE,
                action=OperationAction.DELETE,
                target_uri=uri,
                payload={"reason": "batch_delete"},
            )
        )

    result = client.runtime.transaction.committer.commit("u1", operations)

    assert {item.operation_id for item in result.operations} == {
        "op_batch_delete_0",
        "op_batch_delete_1",
    }
    assert all(item.payload["projection_tombstone_ids"] for item in result.operations)
    for uri, row_id in zip(uris, vector_ids, strict=True):
        assert client.runtime.stores.source.read_object(uri).lifecycle_state.value == "deleted"
        assert index.get_catalog_by_uri(uri, tenant_id="default") == []
        assert index.search_catalog(
            "batch tombstone",
            tenant_id="default",
            filters={"tenant_id": "default"},
        ) == []
        assert vectors.get_vector_metadata(row_id) is None
        assert client.runtime.stores.relation.relations_of(uri, tenant_id="default") == []
    with sqlite3.connect(index.path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM context_paths").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM context_tombstones WHERE status = 'APPLIED'").fetchone()[0] == 2


def test_production_committer_rejects_delete_without_tombstone_service(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    index = SQLiteIndexStore(tmp_path / "indexes" / "catalog.sqlite3")
    relations = SQLiteRelationStore(tmp_path / "indexes" / "relations.sqlite3")
    committer = OperationCommitter(source, index, str(tmp_path), relation_store=relations)
    uri = "memoryos://user/u1/resources/no-delete-bypass"
    obj = ContextObject(
        uri=uri,
        context_type=ContextType.RESOURCE,
        title="no delete bypass",
        owner_user_id="u1",
    )
    seed_context_object(source, index, obj, content="must remain searchable")
    operation = ContextOperation(
        user_id="u1",
        context_type=ContextType.RESOURCE,
        action=OperationAction.DELETE,
        target_uri=uri,
        payload={"reason": "must_not_bypass"},
    )

    with pytest.raises(RuntimeError, match="requires ProjectionTombstoneService"):
        committer.commit(operation.user_id, [operation])

    assert source.read_object(uri).lifecycle_state.value == "active"
    assert index.search_catalog(
        "must remain searchable",
        tenant_id="default",
        filters={"tenant_id": "default"},
    )
    assert not committer.redo.pending_entries()


def test_resume_cannot_substitute_an_unrelated_projection_tombstone(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path), vector_store=InMemoryVectorStore())
    index = cast(SQLiteIndexStore, client.runtime.stores.index)
    uri_a = "memoryos://user/u1/resources/resume-target-a"
    uri_b = "memoryos://user/u1/resources/resume-target-b"
    for uri in (uri_a, uri_b):
        _seed_client_context(
            client,
            ContextObject(
                uri=uri,
                context_type=ContextType.RESOURCE,
                title=uri.rsplit("/", 1)[-1],
                tenant_id="default",
                owner_user_id="u1",
                metadata={"tree_paths": ["resources/desktop"], "source_kind": "resource"},
            ),
            content=f"serving record for {uri}",
        )
    service = client.runtime.context.lifecycle_service.tombstone_service
    unrelated_ids = service.enqueue_uri(
        uri_b,
        tenant_id="default",
        reason="unrelated_cleanup",
        require_source_retired=False,
    )
    forged = ContextOperation(
        operation_id="op_resume_tombstone_substitution",
        user_id="u1",
        context_type=ContextType.RESOURCE,
        action=OperationAction.DELETE,
        target_uri=uri_a,
        payload={
            "tenant_id": "default",
            "reason": "forged_resume",
            "projection_tombstone_ids": list(unrelated_ids),
        },
    )
    manifest = client.runtime.transaction.committer._build_regular_relation_manifest(forged)

    with pytest.raises(RedoIntegrityError, match="exactly one durable"):
        client.runtime.transaction.committer.resume("u1", forged, "started", relation_manifest=manifest)

    durable = ContextOperation.from_dict(forged.to_dict())
    durable.payload.pop("projection_tombstone_ids")
    durable_manifest = client.runtime.transaction.committer._build_regular_relation_manifest(durable)
    client.runtime.transaction.committer.redo.begin(durable, phase="started", relation_manifest=durable_manifest)
    with pytest.raises(RedoIntegrityError, match="does not match its durable entry"):
        client.runtime.transaction.committer.resume("u1", forged, "started", relation_manifest=manifest)

    assert client.runtime.stores.source.read_object(uri_a).lifecycle_state.value == "active"
    assert client.runtime.stores.source.read_object(uri_b).lifecycle_state.value == "active"
    assert index.get_catalog_by_uri(uri_a, tenant_id="default")
    assert index.get_catalog_by_uri(uri_b, tenant_id="default")


def test_successful_delete_with_relations_is_idempotently_replayable(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path), vector_store=InMemoryVectorStore())
    index = cast(SQLiteIndexStore, client.runtime.stores.index)
    uri = "memoryos://user/u1/resources/relation-delete-retry"
    _seed_client_context(
        client,
        ContextObject(
            uri=uri,
            context_type=ContextType.RESOURCE,
            title="relation delete retry",
            tenant_id="default",
            owner_user_id="u1",
            metadata={"tree_paths": ["resources/desktop"], "source_kind": "resource"},
        ),
        content="relation-backed delete retry",
    )
    record = index.get_catalog_by_uri(uri, tenant_id="default")[0]
    relation = ContextRelation(
        source_uri=uri,
        relation_type="references",
        target_uri="memoryos://resources/relation-delete-target",
        metadata={
            "tenant_id": "default",
            "owner_user_id": "u1",
            "catalog_record_key": record.record_key,
        },
    )
    stored = client.runtime.stores.source.read_object(uri)
    stored.relations = [relation]
    client.runtime.stores.source.write_object(stored, content="relation-backed delete retry")
    client.runtime.stores.relation.add_relation(relation, tenant_id="default")
    operation = ContextOperation(
        operation_id="op_relation_delete_retry",
        user_id="u1",
        context_type=ContextType.RESOURCE,
        action=OperationAction.DELETE,
        target_uri=uri,
        payload={"reason": "idempotent_relation_delete"},
    )
    retry_payload = operation.to_dict()

    first = _commit_client_operation(client, operation)
    second = _commit_client_operation(client, ContextOperation.from_dict(retry_payload))

    assert [item.operation_id for item in first.operations] == ["op_relation_delete_retry"]
    assert [item.operation_id for item in second.operations] == ["op_relation_delete_retry"]
    assert client.runtime.stores.relation.relations_of(uri, tenant_id="default") == []


def test_normal_commit_cannot_overwrite_an_early_delete_redo_entry(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path), vector_store=InMemoryVectorStore())
    index = cast(SQLiteIndexStore, client.runtime.stores.index)
    uri_a = "memoryos://user/u1/resources/durable-delete-a"
    uri_b = "memoryos://user/u1/resources/forged-delete-b"
    for uri in (uri_a, uri_b):
        _seed_client_context(
            client,
            ContextObject(
                uri=uri,
                context_type=ContextType.RESOURCE,
                title=uri.rsplit("/", 1)[-1],
                tenant_id="default",
                owner_user_id="u1",
                metadata={"tree_paths": ["resources/desktop"], "source_kind": "resource"},
            ),
            content=f"active source for {uri}",
        )
    durable = ContextOperation(
        operation_id="op_early_delete_collision",
        user_id="u1",
        context_type=ContextType.RESOURCE,
        action=OperationAction.DELETE,
        target_uri=uri_a,
        payload={"tenant_id": "default", "reason": "durable_a"},
    )
    manifest = client.runtime.transaction.committer._build_regular_relation_manifest(durable)
    client.runtime.transaction.committer.redo.begin(durable, phase="started", relation_manifest=manifest)
    assert client.runtime.transaction.committer._prepare_delete_tombstones(durable)
    client.runtime.transaction.committer.redo.advance(
        durable,
        phase="tombstones_enqueued",
        relation_manifest=manifest,
    )
    forged = ContextOperation(
        operation_id=durable.operation_id,
        user_id="u1",
        context_type=ContextType.RESOURCE,
        action=OperationAction.DELETE,
        target_uri=uri_b,
        payload={"tenant_id": "default", "reason": "forged_b"},
    )

    with pytest.raises(RedoIntegrityError, match="different durable effect"):
        _commit_client_operation(client, forged)

    assert client.runtime.stores.source.read_object(uri_a).lifecycle_state.value == "active"
    assert client.runtime.stores.source.read_object(uri_b).lifecycle_state.value == "active"
    assert index.get_catalog_by_uri(uri_a, tenant_id="default")
    assert index.get_catalog_by_uri(uri_b, tenant_id="default")
    entry = client.runtime.transaction.committer.redo.pending_entries()[0]
    assert entry.operation.target_uri == uri_a
    assert entry.phase == "tombstones_enqueued"

    recovered = _commit_client_operation(
        client,
        ContextOperation(
            operation_id=durable.operation_id,
            user_id="u1",
            context_type=ContextType.RESOURCE,
            action=OperationAction.DELETE,
            target_uri=uri_a,
            payload={"tenant_id": "default", "reason": "durable_a"},
        ),
    )
    assert [item.operation_id for item in recovered.operations] == [durable.operation_id]
    assert client.runtime.stores.source.read_object(uri_a).lifecycle_state.value == "deleted"
    assert client.runtime.stores.source.read_object(uri_b).lifecycle_state.value == "active"
    assert index.get_catalog_by_uri(uri_a, tenant_id="default") == []
    assert index.get_catalog_by_uri(uri_b, tenant_id="default")


def test_implicit_target_retry_resumes_exact_resolver_bound_redo(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    uri_a = "memoryos://user/u1/resources/implicit-redo-a"
    uri_b = "memoryos://user/u1/resources/implicit-redo-b"
    for uri in (uri_a, uri_b):
        _seed_client_context(
            client,
            ContextObject(
                uri=uri,
                context_type=ContextType.RESOURCE,
                title="before update",
                tenant_id="default",
                owner_user_id="u1",
                metadata={"tree_paths": ["resources/desktop"], "source_kind": "resource"},
            ),
            content="before update",
        )
    updated = client.runtime.stores.source.read_object(uri_a)
    updated.title = "resolver-bound update"
    request = ContextOperation(
        operation_id="op_implicit_target_redo",
        user_id="u1",
        context_type=ContextType.RESOURCE,
        action=OperationAction.UPDATE,
        target_uri=None,
        payload={
            "tenant_id": "default",
            "context_object": updated.to_dict(),
            "content": "resolver-bound update content",
        },
    )
    retry_payload = request.to_dict()
    durable_request = ContextOperation.from_dict(request.to_dict())
    client.runtime.transaction.committer._validate_and_bind_operations("u1", [durable_request])
    resolved = client.runtime.transaction.committer.target_resolver.resolve(durable_request, user_id="u1")
    assert resolved.resolved and resolved.operation.target_uri == uri_a
    manifest = client.runtime.transaction.committer._build_regular_relation_manifest(resolved.operation)
    client.runtime.transaction.committer.redo.begin(resolved.operation, phase="started", relation_manifest=manifest)

    explicit_substitution = ContextOperation.from_dict(retry_payload)
    explicit_substitution.target_uri = uri_b
    with pytest.raises(RedoIntegrityError, match="different durable effect"):
        _commit_client_operation(client, explicit_substitution)

    recovered = _commit_client_operation(client, ContextOperation.from_dict(retry_payload))

    assert [item.operation_id for item in recovered.operations] == [request.operation_id]
    assert client.runtime.stores.source.read_object(uri_a).title == "resolver-bound update"
    assert client.runtime.stores.source.read_object(uri_b).title == "before update"
    assert not client.runtime.transaction.committer.redo.pending_entries()


def test_fuzzy_delete_retry_returns_durable_committed_target_not_pending(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path), vector_store=InMemoryVectorStore())
    uri_a = "memoryos://user/u1/resources/fuzzy-redo-a"
    uri_b = "memoryos://user/u1/resources/fuzzy-redo-b"
    _seed_client_context(
        client,
        ContextObject(
            uri=uri_a,
            context_type=ContextType.RESOURCE,
            title="uniquefuzzydeleteme",
            tenant_id="default",
            owner_user_id="u1",
            metadata={"tree_paths": ["resources/desktop"], "source_kind": "resource"},
        ),
        content="uniquefuzzydeleteme",
    )
    _seed_client_context(
        client,
        ContextObject(
            uri=uri_b,
            context_type=ContextType.RESOURCE,
            title="unrelated retained object",
            tenant_id="default",
            owner_user_id="u1",
            metadata={"tree_paths": ["resources/desktop"], "source_kind": "resource"},
        ),
        content="unrelated retained object",
    )
    request = ContextOperation(
        operation_id="op_fuzzy_target_delete_redo",
        user_id="u1",
        context_type=ContextType.RESOURCE,
        action=OperationAction.DELETE,
        target_uri=None,
        payload={
            "tenant_id": "default",
            "query": "uniquefuzzydeleteme",
            "reason": "fuzzy redo regression",
        },
    )
    retry_payload = request.to_dict()
    durable_request = ContextOperation.from_dict(request.to_dict())
    client.runtime.transaction.committer._validate_and_bind_operations("u1", [durable_request])
    resolved = client.runtime.transaction.committer.target_resolver.resolve(durable_request, user_id="u1")
    assert resolved.resolved and resolved.operation.target_uri == uri_a
    manifest = client.runtime.transaction.committer._build_regular_relation_manifest(resolved.operation)
    client.runtime.transaction.committer.redo.begin(resolved.operation, phase="started", relation_manifest=manifest)
    assert client.runtime.transaction.committer._prepare_delete_tombstones(resolved.operation)
    client.runtime.transaction.committer.redo.advance(
        resolved.operation,
        phase="tombstones_enqueued",
        relation_manifest=manifest,
    )

    recovered = _commit_client_operation(client, ContextOperation.from_dict(retry_payload))

    assert [item.operation_id for item in recovered.operations] == [request.operation_id]
    assert recovered.pending_operations == []
    assert recovered.operations[0].target_uri == uri_a
    assert client.runtime.stores.source.read_object(uri_a).lifecycle_state.value == "deleted"
    assert client.runtime.stores.source.read_object(uri_b).lifecycle_state.value == "active"
    assert not client.runtime.transaction.committer.redo.pending_entries()


def test_failed_commit_delete_tombstone_replays_exact_binding_on_retry(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    vectors = InMemoryVectorStore()
    client = MemoryOSClient(str(tmp_path), vector_store=vectors)
    index = cast(SQLiteIndexStore, client.runtime.stores.index)
    uri = "memoryos://user/u1/resources/retry-report"
    obj = ContextObject(
        uri=uri,
        context_type=ContextType.RESOURCE,
        title="retry report",
        tenant_id="default",
        owner_user_id="u1",
        metadata={"tree_paths": ["resources/desktop"], "source_kind": "resource"},
    )
    _seed_client_context(client, obj, content="retry tombstone marker")
    record = index.get_catalog_by_uri(uri, tenant_id="default")[0]
    row_id = vector_row_id("default", record.record_key)
    vectors.upsert_vector(
        row_id,
        [1.0, 0.0],
        metadata={
            "tenant_id": "default",
            "catalog_record_key": record.record_key,
            "source_revision": record.source_revision,
            "projection_effect_hash": record.projection_effect_hash,
        },
    )
    operation = ContextOperation(
        operation_id="op_retryable_projection_delete",
        user_id="u1",
        context_type=ContextType.RESOURCE,
        action=OperationAction.DELETE,
        target_uri=uri,
        payload={"reason": "retryable_delete"},
    )
    retry_payload = operation.to_dict()
    original_delete = vectors.delete_vector

    def fail_delete(_row_id: str) -> None:
        raise RuntimeError("vector backend unavailable")

    monkeypatch.setattr(vectors, "delete_vector", fail_delete)
    with pytest.raises(RuntimeError, match="retryable but incomplete"):
        _commit_client_operation(client, operation)

    assert client.runtime.stores.source.read_object(uri).lifecycle_state.value == "deleted"
    assert index.get_catalog_by_uri(uri, tenant_id="default") == []
    assert vectors.get_vector_metadata(row_id) is not None
    with sqlite3.connect(index.path) as conn:
        row = conn.execute(
            "SELECT tombstone_id, status, retry_count FROM context_tombstones WHERE uri = ?",
            (uri,),
        ).fetchone()
    assert row is not None
    tombstone_id = str(row[0])
    assert row[1] == "CLEANING"
    assert int(row[2]) >= 1

    monkeypatch.setattr(vectors, "delete_vector", original_delete)
    retried = _commit_client_operation(client, ContextOperation.from_dict(retry_payload))

    committed = next(item for item in retried.operations if item.operation_id == "op_retryable_projection_delete")
    assert committed.payload["projection_tombstone_ids"] == [tombstone_id]
    assert vectors.get_vector_metadata(row_id) is None
    with sqlite3.connect(index.path) as conn:
        assert (
            conn.execute(
                "SELECT status FROM context_tombstones WHERE tombstone_id = ?",
                (tombstone_id,),
            ).fetchone()[0]
            == "APPLIED"
        )


def test_startup_replays_committed_pending_delete_tombstone(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    vectors = InMemoryVectorStore()
    client = MemoryOSClient(str(tmp_path), vector_store=vectors)
    index = cast(SQLiteIndexStore, client.runtime.stores.index)
    uri = "memoryos://user/u1/resources/startup-replay-report"
    obj = ContextObject(
        uri=uri,
        context_type=ContextType.RESOURCE,
        title="startup replay report",
        tenant_id="default",
        owner_user_id="u1",
        metadata={"tree_paths": ["resources/desktop"], "source_kind": "resource"},
    )
    _seed_client_context(client, obj, content="startup replay tombstone marker")
    record = index.get_catalog_by_uri(uri, tenant_id="default")[0]
    row_id = vector_row_id("default", record.record_key)
    vectors.upsert_vector(
        row_id,
        [1.0, 0.0],
        metadata={
            "tenant_id": "default",
            "catalog_record_key": record.record_key,
            "source_revision": record.source_revision,
            "projection_effect_hash": record.projection_effect_hash,
        },
    )
    operation = ContextOperation(
        operation_id="op_startup_projection_delete",
        user_id="u1",
        context_type=ContextType.RESOURCE,
        action=OperationAction.DELETE,
        target_uri=uri,
        payload={"reason": "startup_retryable_delete"},
    )
    original_delete = vectors.delete_vector
    monkeypatch.setattr(
        vectors,
        "delete_vector",
        lambda _row_id: (_ for _ in ()).throw(RuntimeError("vector backend unavailable")),
    )
    with pytest.raises(RuntimeError, match="retryable but incomplete"):
        _commit_client_operation(client, operation)
    monkeypatch.setattr(vectors, "delete_vector", original_delete)

    restarted = MemoryOSClient(str(tmp_path), vector_store=vectors)

    assert restarted.runtime.readiness.snapshot()["ready"] is True
    assert vectors.get_vector_metadata(row_id) is None
    assert restarted.runtime.readiness.details["generic_tombstones"] == {
        "processed": 1,
        "stale": 0,
    }
    with sqlite3.connect(index.path) as conn:
        assert (
            conn.execute(
                "SELECT status FROM context_tombstones WHERE uri = ?",
                (uri,),
            ).fetchone()[0]
            == "APPLIED"
        )


def test_tombstone_relation_cleanup_filters_ownership_before_bounded_limit(tmp_path) -> None:  # noqa: ANN001
    index = SQLiteIndexStore(tmp_path / "indexes" / "catalog.sqlite3")
    relations = SQLiteRelationStore(tmp_path / "indexes" / "relations.sqlite3")
    timestamp = "2026-07-14T03:00:00+00:00"
    uri = "memoryos://user/u1/resources/owned-report"
    record = CatalogRecord(
        record_key="resource:owned-report",
        uri=uri,
        tenant_id="tenant-a",
        owner_user_id="u1",
        context_type="resource",
        source_kind="resource",
        record_kind=CatalogRecordKind.CONTEXT.value,
        tree_paths=("resources/desktop",),
        created_at=timestamp,
        updated_at=timestamp,
        event_time=timestamp,
        ingested_at=timestamp,
        transaction_time=timestamp,
        title="owned report",
        l1_text="owned relation cleanup",
        source_uri=uri,
        source_digest="owned-digest",
        source_revision=1,
    )
    index.upsert_catalog(record, tenant_id="tenant-a")
    for ordinal in range(1_005):
        relations.add_relation(
            ContextRelation(
                source_uri=uri,
                relation_type=f"unrelated_{ordinal:04d}",
                target_uri=f"memoryos://resources/unrelated-{ordinal:04d}",
                weight=1.0,
                metadata={
                    "tenant_id": "tenant-a",
                    "catalog_record_key": f"other:{ordinal:04d}",
                },
            ),
            tenant_id="tenant-a",
        )
    owned = ContextRelation(
        source_uri=uri,
        relation_type="owned",
        target_uri="memoryos://resources/owned-target",
        weight=0.0,
        metadata={
            "tenant_id": "tenant-a",
            "catalog_record_key": record.record_key,
        },
    )
    relations.add_relation(owned, tenant_id="tenant-a")
    service = ProjectionTombstoneService(index, relation_store=relations)
    tombstones = service.enqueue_uri(
        uri,
        tenant_id="tenant-a",
        reason="ownership-regression",
        require_source_retired=False,
    )

    result = service.process_tombstones(tombstones, tenant_id="tenant-a")

    assert result.processed == tombstones
    retained = relations.relations_of(uri, tenant_id="tenant-a")
    assert len(retained) == 1_005
    assert all(item.metadata["catalog_record_key"].startswith("other:") for item in retained)
    assert owned not in retained


def test_stale_tombstone_never_deletes_newer_catalog_vector_or_relation(tmp_path) -> None:  # noqa: ANN001
    index = SQLiteIndexStore(tmp_path / "indexes" / "catalog.sqlite3")
    relations = SQLiteRelationStore(tmp_path / "indexes" / "relations.sqlite3")
    vectors = InMemoryVectorStore()
    timestamp = "2026-07-14T03:00:00+00:00"
    uri = "memoryos://user/u1/resources/revisioned"
    old = CatalogRecord(
        record_key="resource:revisioned",
        uri=uri,
        tenant_id="tenant-a",
        owner_user_id="u1",
        context_type="resource",
        source_kind="resource",
        record_kind=CatalogRecordKind.CONTEXT.value,
        tree_paths=("resources/desktop",),
        created_at=timestamp,
        updated_at=timestamp,
        event_time=timestamp,
        ingested_at=timestamp,
        transaction_time=timestamp,
        title="old revision",
        l1_text="old revision",
        source_uri=uri,
        source_digest="a" * 64,
        source_revision=1,
    )
    index.upsert_catalog(old, tenant_id="tenant-a")
    row_id = vector_row_id("tenant-a", old.record_key)
    vectors.upsert_vector(
        row_id,
        [1.0, 0.0],
        metadata={
            "tenant_id": "tenant-a",
            "catalog_record_key": old.record_key,
            "source_revision": 1,
            "projection_effect_hash": old.projection_effect_hash,
        },
    )
    relations.add_relation(
        ContextRelation(
            source_uri=uri,
            relation_type="references",
            target_uri="memoryos://resources/shared",
            metadata={"tenant_id": "tenant-a", "catalog_record_key": old.record_key},
        ),
        tenant_id="tenant-a",
    )
    service = ProjectionTombstoneService(index, vector_store=vectors, relation_store=relations)
    tombstones = service.enqueue_uri(
        uri,
        tenant_id="tenant-a",
        reason="old-revision-delete",
        require_source_retired=False,
    )

    newer = replace(
        old,
        updated_at="2026-07-14T04:00:00+00:00",
        transaction_time="2026-07-14T04:00:00+00:00",
        title="new revision",
        l1_text="new revision",
        source_digest="b" * 64,
        source_revision=2,
    )
    index.upsert_catalog(newer, tenant_id="tenant-a")
    vectors.upsert_vector(
        row_id,
        [0.0, 1.0],
        metadata={
            "tenant_id": "tenant-a",
            "catalog_record_key": newer.record_key,
            "source_revision": 2,
            "projection_effect_hash": newer.projection_effect_hash,
        },
    )
    relations.add_relation(
        ContextRelation(
            source_uri=uri,
            relation_type="references",
            target_uri="memoryos://resources/shared",
            metadata={"tenant_id": "tenant-a", "catalog_record_key": newer.record_key},
        ),
        tenant_id="tenant-a",
    )

    result = service.process_tombstones(tombstones, tenant_id="tenant-a")

    assert result.stale == tombstones
    retained = index.get_catalog(newer.record_key, tenant_id="tenant-a")
    assert retained is not None
    assert retained.record_key == newer.record_key
    assert retained.source_revision == 2
    assert retained.source_digest == newer.source_digest
    assert vectors.get_vector_metadata(row_id) == {
        "tenant_id": "tenant-a",
        "catalog_record_key": newer.record_key,
        "source_revision": 2,
        "projection_effect_hash": newer.projection_effect_hash,
    }
    assert relations.relations_of(uri, tenant_id="tenant-a")[0].metadata["catalog_record_key"] == newer.record_key


def test_old_record_key_cleanup_preserves_vector_owned_by_new_manifest(tmp_path) -> None:  # noqa: ANN001
    index = SQLiteIndexStore(tmp_path / "indexes" / "catalog.sqlite3")
    vectors = InMemoryVectorStore()
    timestamp = "2026-07-14T03:00:00+00:00"
    uri = "memoryos://user/u1/sessions/history/s1/context/root"
    old = CatalogRecord(
        record_key="session:s1:manifest:old:root",
        uri=uri,
        tenant_id="tenant-a",
        owner_user_id="u1",
        session_id="s1",
        context_type="session",
        source_kind="session_root",
        record_kind=CatalogRecordKind.SESSION_ROOT.value,
        tree_paths=("sessions/s1",),
        created_at=timestamp,
        updated_at=timestamp,
        event_time=timestamp,
        ingested_at=timestamp,
        transaction_time=timestamp,
        title="old manifest",
        l1_text="old manifest",
        source_uri="memoryos://user/u1/sessions/history/s1",
        source_digest="a" * 64,
    )
    index.upsert_catalog(old, tenant_id="tenant-a")
    old_row_id = vector_row_id("tenant-a", old.record_key)
    vectors.upsert_vector(
        old_row_id,
        [1.0, 0.0],
        metadata={"tenant_id": "tenant-a", "catalog_record_key": old.record_key},
    )
    service = ProjectionTombstoneService(index, vector_store=vectors)
    tombstones = service.enqueue_uri(
        uri,
        tenant_id="tenant-a",
        reason="old-manifest-retired",
        require_source_retired=False,
    )
    newer = replace(
        old,
        record_key="session:s1:manifest:new:root",
        updated_at="2026-07-14T04:00:00+00:00",
        transaction_time="2026-07-14T04:00:00+00:00",
        title="new manifest",
        l1_text="new manifest",
        source_digest="b" * 64,
    )
    index.upsert_catalog(newer, tenant_id="tenant-a")
    new_row_id = vector_row_id("tenant-a", newer.record_key)
    vectors.upsert_vector(
        new_row_id,
        [0.0, 1.0],
        metadata={"tenant_id": "tenant-a", "catalog_record_key": newer.record_key},
    )

    result = service.process_tombstones(tombstones, tenant_id="tenant-a")

    assert result.processed == tombstones
    assert index.get_catalog(old.record_key, tenant_id="tenant-a") is None
    retained = index.get_catalog(newer.record_key, tenant_id="tenant-a")
    assert retained is not None
    assert retained.record_key == newer.record_key
    assert retained.source_digest == newer.source_digest
    assert vectors.get_vector_metadata(old_row_id) is None
    assert vectors.get_vector_metadata(new_row_id) == {
        "tenant_id": "tenant-a",
        "catalog_record_key": newer.record_key,
    }
