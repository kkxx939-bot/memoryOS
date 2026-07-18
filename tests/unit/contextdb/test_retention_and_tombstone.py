from __future__ import annotations

from datetime import datetime, timezone

from memoryos.contextdb.catalog import CatalogRecord, CatalogRecordKind, ServingTier
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retention import CatalogRetentionManager, RetentionPolicy
from memoryos.contextdb.store.local_stores import FileSystemSourceStore
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore
from memoryos.contextdb.store.sqlite_relation_store import SQLiteRelationStore
from memoryos.contextdb.store.vector_store import InMemoryVectorStore, vector_row_id
from memoryos.contextdb.tombstone import ProjectionTombstoneService


def test_tombstone_replay_is_tenant_scoped_and_waits_for_source_retirement(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path / "source", tenant_id="tenant-a")
    index = SQLiteIndexStore(tmp_path / "catalog.sqlite3")
    relations = SQLiteRelationStore(tmp_path / "relations.sqlite3")
    vectors = InMemoryVectorStore()
    obj = ContextObject(
        uri="memoryos://user/alice/resources/report",
        context_type=ContextType.RESOURCE,
        title="quarterly report",
        tenant_id="tenant-a",
        owner_user_id="alice",
    )
    source.write_object(obj, content="quarterly revenue")
    index.upsert_index(obj, content="quarterly revenue", tenant_id="tenant-a")
    record = index.get_catalog_by_uri(obj.uri, tenant_id="tenant-a")[0]
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
            source_uri=obj.uri,
            relation_type="references",
            target_uri="memoryos://resources/shared",
            metadata={
                "tenant_id": "tenant-a",
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
        obj.uri,
        tenant_id="tenant-a",
        reason="resource retired",
    )

    assert service.process_pending(tenant_id="tenant-a").failed == tombstones
    assert index.get_catalog(record.record_key, tenant_id="tenant-a") is not None
    source.soft_delete(obj.uri, "resource retired")
    assert service.process_pending(tenant_id="tenant-a").processed == tombstones
    assert index.get_catalog(record.record_key, tenant_id="tenant-a") is None
    assert vectors.get_vector_metadata(row_id) is None
    assert relations.relations_of(obj.uri, tenant_id="tenant-a") == []


def _record(*, record_key: str, kind: str, document_id: str = "") -> CatalogRecord:
    timestamp = "2026-01-01T00:00:00+00:00"
    return CatalogRecord(
        record_key=record_key,
        uri=f"memoryos://user/alice/catalog/{record_key}",
        tenant_id="tenant-a",
        owner_user_id="alice",
        session_id="session-a" if not document_id else "",
        context_type="memory" if document_id else "session",
        source_kind="markdown_memory_document" if document_id else "session_root",
        record_kind=kind,
        tree_paths=("memories/profile",) if document_id else ("sessions/session-a",),
        created_at=timestamp,
        updated_at=timestamp,
        event_time=timestamp,
        ingested_at=timestamp,
        transaction_time=timestamp,
        title=record_key,
        l1_text=record_key,
        source_uri=f"memoryos://user/alice/source/{record_key}",
        source_digest="a" * 64,
        document_id=document_id,
        document_kind="profile" if document_id else "",
        document_revision=1 if document_id else 0,
        projection_generation=1 if document_id else 0,
        serving_tier=ServingTier.HOT.value,
        metadata={
            "vector_eligible": True,
            **({"relative_path": "profile.md"} if document_id else {}),
        },
    )


def test_retention_leaves_markdown_document_state_to_its_projector(tmp_path) -> None:
    index = SQLiteIndexStore(tmp_path / "catalog.sqlite3")
    vectors = InMemoryVectorStore()
    document = _record(
        record_key="memory-document:alice:profile",
        kind=CatalogRecordKind.MEMORY_DOCUMENT.value,
        document_id="memdoc_0123456789abcdef01234567",
    )
    session = _record(
        record_key="session:session-a:root",
        kind=CatalogRecordKind.SESSION_ROOT.value,
    )
    index.replace_memory_document_projection(
        document,
        (),
        0,
        tenant_id="tenant-a",
        owner_user_id="alice",
    )
    index.upsert_catalog(session, tenant_id="tenant-a")
    for record in (document, session):
        vectors.upsert_vector(
            vector_row_id("tenant-a", record.record_key),
            [1.0, 0.0],
            metadata={
                "tenant_id": "tenant-a",
                "catalog_record_key": record.record_key,
                "source_revision": record.source_revision,
                "projection_effect_hash": record.projection_effect_hash,
            },
        )
    manager = CatalogRetentionManager(
        index,
        vector_store=vectors,
        policy=RetentionPolicy.from_config(
            {"hot_days": 1, "warm_days": 2, "cold_days": 3}
        ),
    )

    tiers = manager.apply_serving_tiers(
        tenant_id="tenant-a",
        now=datetime(2026, 7, 18, tzinfo=timezone.utc),
    )
    vectors_result = manager.gc_vectors(tenant_id="tenant-a")

    current_document = index.get_catalog(document.record_key, tenant_id="tenant-a")
    current_session = index.get_catalog(session.record_key, tenant_id="tenant-a")
    assert current_document is not None and current_document.serving_tier == ServingTier.HOT.value
    assert current_session is not None and current_session.serving_tier == ServingTier.ARCHIVED.value
    assert tiers.tier_changes == 1
    assert vectors_result.vectors_deleted == 1
    assert vectors.get_vector_metadata(vector_row_id("tenant-a", document.record_key)) is not None
    assert vectors.get_vector_metadata(vector_row_id("tenant-a", session.record_key)) is None
