from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pytest

from memoryos.contextdb.catalog import CatalogRecord, CatalogRecordKind, ServingTier
from memoryos.contextdb.retention import CatalogRetentionManager, RetentionPolicy
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore
from memoryos.contextdb.store.vector_store import InMemoryVectorStore
from memoryos.providers.embedding import HashingEmbeddingProvider
from memoryos.runtime import RuntimeConfig, build_runtime_container


def _runtime_archive() -> SessionArchive:
    return SessionArchive(
        user_id="u1",
        session_id="runtime-vector-session",
        archive_uri="memoryos://user/u1/sessions/history/runtime-vector-session",
        created_at="2026-06-01T09:00:00+08:00",
        metadata={
            "tenant_id": "tenant-a",
            "timezone": "Asia/Singapore",
            "project_id": "memoryOS",
        },
        messages=[
            {
                "role": "user",
                "content": "read the desktop release plan",
                "occurred_at": "2026-06-01T09:00:00+08:00",
            }
        ],
        tool_results=[
            {
                "tool_name": "read_file",
                "output": "API_KEY=runtime-secret release details",
                "path": "/Users/u1/Desktop/release-plan.md",
                "important": True,
                "occurred_at": "2026-06-01T09:01:00+08:00",
            }
        ],
        observations=[
            {
                "raw_text": "reusable release observation",
                "important": True,
                "occurred_at": "2026-06-01T09:02:00+08:00",
            }
        ],
    )


def test_runtime_projects_safe_session_vectors_and_retention_replays_tombstones(tmp_path) -> None:
    vectors = InMemoryVectorStore()
    runtime = build_runtime_container(
        RuntimeConfig(
            root=str(tmp_path),
            tenant_id="tenant-a",
            retrieval={"vectorize_important_session_events": True},
            retention={
                "hot_days": 1,
                "warm_days": 2,
                "cold_days": 3,
                "batch_size": 32,
            },
        ),
        vector_store=vectors,
        embedding_provider=HashingEmbeddingProvider(),
    )
    archive = _runtime_archive()

    committed = runtime.session_commit_service.sync_archive(archive, enqueue_commit_job=False)

    assert committed.session_projection_status == "projected"
    assert runtime.retention_manager is not None
    assert runtime.context_db.retention_manager is runtime.retention_manager
    records = runtime.index_store.scan_catalog_batch(  # type: ignore[attr-defined]
        filters={
            "tenant_id": "tenant-a",
            "session_ids": (archive.session_id,),
            "include_inactive": True,
        },
        limit=100,
    )
    vector_kinds = {str((vectors.get_vector_metadata(uri) or {}).get("record_kind") or "") for uri in vectors.rows}
    assert CatalogRecordKind.SESSION_ROOT.value in vector_kinds
    assert CatalogRecordKind.SEMANTIC_SEGMENT.value in vector_kinds
    assert CatalogRecordKind.OBSERVATION.value in vector_kinds
    assert CatalogRecordKind.RESOURCE_REFERENCE.value in vector_kinds
    assert CatalogRecordKind.MESSAGE.value not in vector_kinds
    assert CatalogRecordKind.TOOL_RESULT.value not in vector_kinds
    initial_vector_count = len(vectors.rows)

    cycle = runtime.context_db.run_retention_cycle(now=datetime(2026, 7, 14, 12, tzinfo=timezone.utc))

    assert cycle["tiers"]["tier_changes"] == len(records)
    assert cycle["vectors"]["vectors_deleted"] == initial_vector_count
    assert not vectors.rows
    retained = runtime.index_store.scan_catalog_batch(  # type: ignore[attr-defined]
        filters={
            "tenant_id": "tenant-a",
            "session_ids": (archive.session_id,),
            "include_inactive": True,
        },
        limit=100,
    )
    assert retained and all(record.serving_tier == ServingTier.ARCHIVED.value for record in retained)
    evidence = runtime.session_archive_store.read_archive(archive.archive_uri)
    assert evidence.archive_digest == archive.archive_digest
    with sqlite3.connect(runtime.index_store.path) as connection:  # type: ignore[attr-defined]
        rows = connection.execute(
            "SELECT status, payload_json FROM context_tombstones "
            "WHERE json_extract(payload_json, '$.projection_action') = 'vector_delete'"
        ).fetchall()
    assert len(rows) == initial_vector_count
    assert all(status == "APPLIED" for status, _payload in rows)
    assert all("runtime-secret" not in str(payload) for _status, payload in rows)

    root_record = next(record for record in retained if record.record_kind == CatalogRecordKind.SESSION_ROOT.value)
    restored = runtime.context_db.restore_cold_context(
        root_record.record_key,
        now=datetime(2026, 7, 14, 12, tzinfo=timezone.utc),
    )
    assert restored["serving_tier"] == ServingTier.WARM.value
    timeline = runtime.context_db.compact_timeline_context(
        "timeline/2026/06/01",
        owner_user_id="u1",
        now=datetime(2026, 7, 14, 12, tzinfo=timezone.utc),
    )
    assert timeline is not None
    assert timeline["record_kind"] == CatalogRecordKind.TREE_OVERVIEW.value


def test_session_vector_failure_keeps_durable_job_and_replays_idempotently(tmp_path) -> None:
    class FailSecondVectorWrite(InMemoryVectorStore):
        writes = 0
        failed = False

        def upsert_vector(self, uri: str, embedding: list[float], metadata: dict | None = None) -> None:
            self.writes += 1
            if self.writes == 2 and not self.failed:
                self.failed = True
                raise RuntimeError("temporary vector outage Authorization: Token do-not-trace")
            super().upsert_vector(uri, embedding, metadata)

    vectors = FailSecondVectorWrite()
    runtime = build_runtime_container(
        RuntimeConfig(
            root=str(tmp_path),
            tenant_id="tenant-a",
            retrieval={"vectorize_important_session_events": True},
        ),
        vector_store=vectors,
        embedding_provider=HashingEmbeddingProvider(),
    )
    archive = _runtime_archive()

    with pytest.raises(RuntimeError, match="temporary vector outage"):
        runtime.session_commit_service.sync_archive(archive)

    queued = runtime.queue_store.get(archive.task_id)
    assert queued is not None and queued.status == "pending"
    degraded = runtime.index_store.scan_catalog_batch(  # type: ignore[attr-defined]
        filters={
            "tenant_id": "tenant-a",
            "session_ids": (archive.session_id,),
            "include_inactive": True,
        },
        limit=100,
    )
    assert degraded and {record.projection_status for record in degraded} == {"DEGRADED"}
    assert not vectors.rows, "partial vector writes must be removed before replay"
    assert "do-not-trace" not in repr([record.metadata for record in degraded])

    leased = runtime.queue_store.lease("session_commit", lease_owner="test-replay", limit=1)[0]
    runtime.session_commit_service.async_commit(archive)
    runtime.queue_store.ack(leased)

    projected = runtime.index_store.scan_catalog_batch(  # type: ignore[attr-defined]
        filters={
            "tenant_id": "tenant-a",
            "session_ids": (archive.session_id,),
            "include_inactive": True,
        },
        limit=100,
    )
    assert projected and {record.projection_status for record in projected} == {"PROJECTED"}
    assert vectors.rows
    assert runtime.queue_store.get(archive.task_id).status == "done"  # type: ignore[union-attr]


def test_retention_vector_failure_stays_durable_and_retryable(tmp_path) -> None:
    class FailOnceVectorStore(InMemoryVectorStore):
        should_fail = True

        def delete_vector(self, uri: str) -> None:
            if self.should_fail:
                self.should_fail = False
                raise RuntimeError("temporary vector outage password=retention-secret")
            super().delete_vector(uri)

    catalog = SQLiteIndexStore(tmp_path / "catalog.sqlite3")
    vectors = FailOnceVectorStore()
    timestamp = "2026-01-01T00:00:00+00:00"
    record = CatalogRecord(
        record_key="session:stale:root",
        uri="memoryos://user/u1/sessions/history/stale/context/root",
        tenant_id="tenant-a",
        owner_user_id="u1",
        session_id="stale",
        context_type="session",
        source_kind="session_root",
        record_kind=CatalogRecordKind.SESSION_ROOT.value,
        tree_paths=("sessions/stale",),
        created_at=timestamp,
        updated_at=timestamp,
        event_time=timestamp,
        ingested_at=timestamp,
        transaction_time=timestamp,
        title="stale session",
        l0_text="stale",
        l1_text="stale session",
        source_uri="memoryos://user/u1/sessions/history/stale",
        source_digest="a" * 64,
        serving_tier=ServingTier.HOT.value,
        metadata={"vector_eligible": True},
    )
    catalog.upsert_catalog(record)
    vectors.upsert_vector(record.uri, [1.0, 0.0])
    manager = CatalogRetentionManager(
        catalog,
        vector_store=vectors,
        policy=RetentionPolicy.from_config({"hot_days": 1, "warm_days": 2, "cold_days": 3}),
    )
    manager.apply_serving_tiers(
        tenant_id="tenant-a",
        now=datetime(2026, 7, 14, tzinfo=timezone.utc),
    )

    with pytest.raises(RuntimeError, match="durable but incomplete"):
        manager.gc_vectors(tenant_id="tenant-a")

    assert catalog.get_catalog(record.record_key, tenant_id="tenant-a") is not None
    assert record.uri in vectors.rows
    with sqlite3.connect(catalog.path) as connection:
        failed = connection.execute(
            "SELECT status, retry_count, last_error, payload_json FROM context_tombstones "
            "WHERE json_extract(payload_json, '$.projection_action') = 'vector_delete'"
        ).fetchone()
    # Catalog deletion ownership remains durable while the external Vector
    # cleanup is retried; allowing a new projection during this window would
    # let the retry delete the replacement Vector.
    assert failed is not None and failed[0] == "CLEANING" and failed[1] == 1
    assert "retention-secret" not in failed[2]
    assert json.loads(failed[3])["catalog_record_key"] == record.record_key

    retried = manager.gc_vectors(tenant_id="tenant-a")

    assert retried.vectors_deleted == 1
    assert record.uri not in vectors.rows
    assert catalog.get_catalog(record.record_key, tenant_id="tenant-a") is not None


def test_server_runtime_rejects_local_vector_alias_as_production_backend(tmp_path) -> None:
    with pytest.raises(ValueError, match="production VectorStore"):
        build_runtime_container(
            RuntimeConfig(root=str(tmp_path), mode="server"),
            vector_store=InMemoryVectorStore(),
            embedding_provider=HashingEmbeddingProvider(),
        )
