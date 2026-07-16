from __future__ import annotations

import sqlite3
import threading
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from memoryos.contextdb.catalog import (
    CatalogProjectionStatus,
    CatalogRecord,
    CatalogRecordKind,
    ServingTier,
)
from memoryos.contextdb.retention import CatalogRetentionManager, RetentionPolicy
from memoryos.contextdb.session.context_projector import SessionContextProjector
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.session.session_commit import SessionCommitService
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.local_stores import InMemoryQueueStore
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore
from memoryos.contextdb.store.sqlite_lock_store import SQLiteLockStore
from memoryos.contextdb.store.vector_store import InMemoryVectorStore
from memoryos.contextdb.unified_migration import (
    MigrationState,
    ReadRoute,
    RuntimeMigrationCoordinator,
    UnifiedContextMigration,
)


def _write_archive(
    archive_store: SessionArchiveStore,
    session_id: str,
    *,
    created_at: str = "2026-07-14T09:00:00+08:00",
) -> SessionArchive:
    archive = SessionArchive(
        user_id="u1",
        session_id=session_id,
        archive_uri=f"memoryos://user/u1/sessions/history/{session_id}",
        created_at=created_at,
        metadata={"tenant_id": "tenant-a", "timezone": "Asia/Singapore"},
        messages=[{"role": "user", "content": f"message {session_id}", "occurred_at": created_at}],
        tool_results=[
            {
                "tool_name": "read_file",
                "output": f"result {session_id}",
                "path": f"/Users/u1/Desktop/{session_id}.txt",
                "occurred_at": created_at,
            }
        ],
    )
    archive_store.write_sync_archive(archive)
    return archive


def _migration(
    catalog: SQLiteIndexStore,
    archives: SessionArchiveStore,
    *,
    batch_size: int = 2,
) -> UnifiedContextMigration:
    return UnifiedContextMigration(
        catalog,
        archives,
        SessionContextProjector(catalog),
        tenant_id="tenant-a",
        batch_size=batch_size,
        minimum_shadow_samples=2,
        maximum_shadow_mismatch_ratio=0.0,
        lock_store=SQLiteLockStore(catalog.path.with_name("migration-locks.sqlite3")),
    )


def _record_matching_shadow_reads(migration: UnifiedContextMigration, count: int) -> None:
    for ordinal in range(count):
        digest = f"matching-shadow-result-{ordinal}"
        recorded = migration.state_store.record_migration_shadow_read(
            migration.migration_name,
            {
                "plan_digest": f"shadow-plan-{ordinal}",
                "legacy_count": 1,
                "unified_count": 1,
                "overlap_count": 1,
                "legacy_digest": digest,
                "unified_digest": digest,
                "matched": False,
            },
            tenant_id=migration.tenant_id,
        )
        assert recorded is not None and recorded["matched"] == 1


def test_migration_backfill_is_batched_resumable_and_cutover_can_rollback(tmp_path) -> None:
    archives = SessionArchiveStore(tmp_path, tenant_id="tenant-a")
    for session_id in ("s1", "s2", "s3"):
        _write_archive(archives, session_id)
    catalog = SQLiteIndexStore(tmp_path / "catalog.sqlite3")
    migration = _migration(catalog, archives)

    assert migration.initialize()["state"] == MigrationState.NOT_STARTED.value
    assert migration.prepare_schema()["state"] == MigrationState.SCHEMA_READY.value
    assert migration.prepare_schema()["state"] == MigrationState.SCHEMA_READY.value
    assert migration.start_backfill()["state"] == MigrationState.BACKFILLING.value

    first = migration.backfill_next_batch()
    assert first.processed_archives == 2
    assert first.projected_records > 0
    assert not first.complete
    assert first.state is MigrationState.BACKFILLING
    assert first.checkpoint.endswith("commit_head.json")

    # A new service instance proves the durable checkpoint resumes instead of
    # loading or reprojecting all SessionArchive objects at once.
    restarted = _migration(SQLiteIndexStore(catalog.path), archives)
    second = restarted.backfill_next_batch()
    assert second.processed_archives == 1
    assert second.complete
    assert second.state is MigrationState.DUAL_WRITE
    assert restarted.feature_gate.dual_write_enabled
    assert restarted.feature_gate.read_route is ReadRoute.SHADOW

    count_after_backfill = len(
        catalog.scan_catalog_batch(
            filters={"tenant_id": "tenant-a", "include_inactive": True},
            limit=1_000,
        )
    )
    replay = restarted.backfill_next_batch()
    assert replay.processed_archives == 0 and replay.complete
    assert (
        len(
            catalog.scan_catalog_batch(
                filters={"tenant_id": "tenant-a", "include_inactive": True},
                limit=1_000,
            )
        )
        == count_after_backfill
    )

    restarted.start_shadow_validation()
    first_shadow = restarted.validate_next_shadow_batch()
    assert first_shadow.sample_count == 2
    assert first_shadow.mismatch_count == 0
    assert not first_shadow.complete
    second_shadow = restarted.validate_next_shadow_batch()
    assert second_shadow.sample_count == 3
    assert second_shadow.mismatch_count == 0
    assert second_shadow.complete
    shadow_state = catalog.get_migration_state(restarted.migration_name, tenant_id="tenant-a")
    assert shadow_state is not None
    epoch = str(shadow_state["details_json"]["shadow_validation_epoch"])
    proofs = catalog.list_migration_equivalence_proofs(
        restarted.migration_name,
        tenant_id="tenant-a",
        validation_epoch=epoch,
        limit=10,
    )
    assert len(proofs) == 3
    assert {proof["plane"] for proof in proofs} == {"session_archive"}
    assert "memoryos://" not in str(proofs)
    _record_matching_shadow_reads(restarted, restarted.minimum_shadow_samples)
    assert restarted.mark_ready_to_cutover()["state"] == MigrationState.READY_TO_CUTOVER.value
    assert restarted.cutover()["state"] == MigrationState.CUTOVER.value
    assert restarted.feature_gate.read_route is ReadRoute.UNIFIED
    assert restarted.feature_gate.legacy_fallback_enabled

    rolled_back = restarted.rollback("bounded shadow regression")
    assert rolled_back["state"] == MigrationState.ROLLBACK.value
    assert restarted.feature_gate.read_route is ReadRoute.LEGACY
    assert restarted.feature_gate.legacy_fallback_enabled

    # Rollback starts a new repair epoch.  Prior checkpoints must not skip a
    # damaged/cleared derived Catalog; immutable Evidence is replayed in the
    # same bounded batches while its Source remains unchanged.
    catalog.clear()
    restarted.start_backfill()
    for _ in range(10):
        batch = restarted.backfill_next_batch()
        if batch.complete:
            break
    else:  # pragma: no cover - bounded migration regression guard.
        raise AssertionError("rollback repair backfill did not complete")
    assert batch.state is MigrationState.DUAL_WRITE
    assert catalog.list_catalog(
        filters={"tenant_id": "tenant-a", "session_ids": ("s1", "s2", "s3")},
        limit=100,
    )
    restarted.start_shadow_validation()
    for _ in range(10):
        if restarted.validate_next_shadow_batch().complete:
            break
    else:  # pragma: no cover - bounded migration regression guard.
        raise AssertionError("shadow validation did not reach its bounded checkpoint")
    _record_matching_shadow_reads(restarted, restarted.minimum_shadow_samples)
    restarted.mark_ready_to_cutover()
    restarted.cutover()
    assert restarted.complete()["state"] == MigrationState.COMPLETED.value
    assert not restarted.feature_gate.legacy_fallback_enabled

    # Migration changes serving projections only; immutable evidence remains
    # readable with its original manifest and digest.
    original = archives.read_archive("memoryos://user/u1/sessions/history/s1")
    assert original.archive_digest
    assert original.messages[0]["content"] == "message s1"


def test_shadow_threshold_and_failed_backfill_are_fail_closed_and_resumable(tmp_path) -> None:
    archives = SessionArchiveStore(tmp_path, tenant_id="tenant-a")
    _write_archive(archives, "s1")
    catalog = SQLiteIndexStore(tmp_path / "catalog.sqlite3")
    migration = _migration(catalog, archives, batch_size=1)
    migration.prepare_schema()
    migration.start_backfill()

    class BrokenProjector:
        def project(self, archive: SessionArchive) -> object:
            raise RuntimeError(f"projection failed for {archive.session_id}: password=secret")

    broken = UnifiedContextMigration(
        catalog,
        archives,
        BrokenProjector(),  # type: ignore[arg-type]
        tenant_id="tenant-a",
        batch_size=1,
        minimum_shadow_samples=2,
    )
    with pytest.raises(RuntimeError, match="projection failed"):
        broken.backfill_next_batch()
    failed = catalog.get_migration_state("unified-context-catalog-v1", tenant_id="tenant-a")
    assert failed is not None and failed["state"] == MigrationState.FAILED.value
    assert "secret" not in failed["last_error"]
    with pytest.raises(ValueError, match="cannot cut over from FAILED"):
        migration.cutover()

    assert migration.resume_failed()["state"] == MigrationState.BACKFILLING.value
    assert migration.backfill_next_batch().state is MigrationState.DUAL_WRITE
    projected = catalog.list_catalog(
        filters={"tenant_id": "tenant-a", "session_ids": ("s1",)},
        limit=100,
    )
    assert projected
    catalog.delete_catalog(projected[0].record_key, tenant_id="tenant-a")
    migration.minimum_shadow_samples = 1
    migration.start_shadow_validation()
    mismatch = migration.validate_next_shadow_batch()
    assert mismatch.sample_count == 1
    assert mismatch.mismatch_count == 1
    with pytest.raises(ValueError, match="mismatch threshold"):
        migration.mark_ready_to_cutover()


def test_backfill_dual_write_closes_earlier_sort_key_checkpoint_race(tmp_path) -> None:
    archives = SessionArchiveStore(tmp_path, tenant_id="tenant-a")
    _write_archive(archives, "s2")
    catalog = SQLiteIndexStore(tmp_path / "catalog.sqlite3")
    delegate = SessionContextProjector(catalog)
    projection_started = threading.Event()
    release_projection = threading.Event()

    class BlockingProjector:
        def project(self, archive: SessionArchive):  # noqa: ANN202
            if archive.session_id == "s2":
                projection_started.set()
                assert release_projection.wait(timeout=5)
            return delegate.project(archive)

    migration = UnifiedContextMigration(
        catalog,
        archives,
        BlockingProjector(),  # type: ignore[arg-type]
        tenant_id="tenant-a",
        batch_size=1,
        minimum_shadow_samples=1,
    )
    migration.prepare_schema()
    migration.start_backfill()
    assert migration.feature_gate.dual_write_enabled

    failures: list[BaseException] = []

    def run_batch() -> None:
        try:
            migration.backfill_next_batch()
        except BaseException as exc:  # pragma: no cover - asserted below.
            failures.append(exc)

    thread = threading.Thread(target=run_batch)
    thread.start()
    assert projection_started.wait(timeout=5)

    # This archive sorts before the already selected s2 checkpoint.  The
    # BACKFILLING dual-write is the only online path that can prevent a miss.
    earlier = _write_archive(archives, "s1")
    service = SessionCommitService(
        archives,
        InMemoryQueueStore(),
        allow_plan_only=True,
        session_projector=delegate,
        migration_gate=RuntimeMigrationCoordinator(
            catalog,
            tenant_id="tenant-a",
            lock_store=SQLiteLockStore(tmp_path / "runtime-migration-locks.sqlite3"),
        ),
    )
    result = service.sync_archive(earlier, enqueue_commit_job=False)
    assert result.session_projection_status == "projected"
    release_projection.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert failures == []

    while migration.state is MigrationState.BACKFILLING:
        migration.backfill_next_batch()
    assert migration.state is MigrationState.DUAL_WRITE
    assert catalog.list_catalog(
        filters={"tenant_id": "tenant-a", "session_ids": ("s1",)},
        limit=100,
    )


def _record(
    record_key: str,
    *,
    event_time: datetime,
    record_kind: CatalogRecordKind = CatalogRecordKind.MESSAGE,
    session_id: str = "s-retention",
    lifecycle_state: str = "active",
    projection_status: CatalogProjectionStatus = CatalogProjectionStatus.PROJECTED,
) -> CatalogRecord:
    timestamp = event_time.astimezone(timezone.utc).isoformat()
    return CatalogRecord(
        record_key=record_key,
        uri=f"memoryos://user/u1/sessions/history/{session_id}/context/{record_key}",
        tenant_id="tenant-a",
        owner_user_id="u1",
        session_id=session_id,
        context_type="session",
        source_kind=record_kind.value,
        record_kind=record_kind.value,
        lifecycle_state=lifecycle_state,
        tree_paths=("sessions/s-retention", "timeline/2026/07/14"),
        created_at=timestamp,
        updated_at=timestamp,
        event_time=timestamp,
        ingested_at=timestamp,
        transaction_time=timestamp,
        title=record_key,
        l0_text=f"abstract {record_key}",
        l1_text=f"overview {record_key}",
        l2_uri=f"memoryos://user/u1/sessions/history/{session_id}",
        source_uri=f"memoryos://user/u1/sessions/history/{session_id}",
        source_digest=(record_key.encode().hex() + "0" * 64)[:64],
        source_revision=1,
        serving_tier=ServingTier.HOT.value,
        projection_status=projection_status.value,
        metadata={"vector_eligible": True},
    )


def test_retention_tiers_compaction_gc_and_cold_restore_preserve_evidence(tmp_path) -> None:
    now = datetime(2026, 7, 14, 12, tzinfo=timezone.utc)
    archives = SessionArchiveStore(tmp_path, tenant_id="tenant-a")
    evidence = _write_archive(archives, "s-retention", created_at=now.isoformat())
    catalog = SQLiteIndexStore(tmp_path / "catalog.sqlite3")
    vector = InMemoryVectorStore()
    policy = RetentionPolicy(
        hot_for=timedelta(days=7),
        warm_for=timedelta(days=30),
        cold_for=timedelta(days=180),
        tombstone_journal_for=timedelta(days=30),
        batch_size=2,
        max_compaction_sources=32,
    )
    manager = CatalogRetentionManager(catalog, vector_store=vector, policy=policy)

    recent = _record("recent", event_time=now - timedelta(days=1))
    warm = _record("warm", event_time=now - timedelta(days=10))
    cold = _record("cold", event_time=now - timedelta(days=60))
    archived = _record("archived", event_time=now - timedelta(days=300))
    current = replace(
        _record(
            "slot:preference:current",
            event_time=now - timedelta(days=300),
            record_kind=CatalogRecordKind.CURRENT_SLOT,
            session_id="",
        ),
        canonical_slot_id="preference",
        canonical_claim_id="claim-current",
    )
    for item in (recent, warm, cold, archived, current):
        catalog.upsert_catalog(item)
        vector.upsert_vector(item.uri, [1.0, 0.0], {"record_key": item.record_key})

    result = manager.apply_serving_tiers(tenant_id="tenant-a", now=now)
    assert result.scanned == 5
    assert catalog.get_catalog(recent.record_key).serving_tier == ServingTier.HOT.value  # type: ignore[union-attr]
    assert catalog.get_catalog(warm.record_key).serving_tier == ServingTier.WARM.value  # type: ignore[union-attr]
    assert catalog.get_catalog(cold.record_key).serving_tier == ServingTier.COLD.value  # type: ignore[union-attr]
    assert catalog.get_catalog(archived.record_key).serving_tier == ServingTier.ARCHIVED.value  # type: ignore[union-attr]
    assert catalog.get_catalog(current.record_key).serving_tier == ServingTier.HOT.value  # type: ignore[union-attr]

    vector_result = manager.gc_vectors(tenant_id="tenant-a")
    assert vector_result.vectors_deleted == 3
    assert recent.uri in vector.rows and current.uri in vector.rows
    assert warm.uri not in vector.rows and cold.uri not in vector.rows and archived.uri not in vector.rows

    restored = manager.restore_cold_record(archived.record_key, tenant_id="tenant-a", now=now)
    assert restored.serving_tier == ServingTier.WARM.value
    assert restored.projection_status == CatalogProjectionStatus.PROJECTED.value
    assert catalog.search_catalog(
        "archived",
        filters={"tenant_id": "tenant-a"},
        limit=5,
    )

    # Compaction creates bounded derived L1/overview records and moves atomic
    # serving rows colder, but never deletes the SessionArchive L2 evidence.
    catalog.upsert_catalog(
        _record(
            "segment",
            event_time=now - timedelta(days=2),
            record_kind=CatalogRecordKind.SEMANTIC_SEGMENT,
        )
    )
    compacted = manager.compact_session(
        tenant_id="tenant-a",
        owner_user_id="u1",
        session_id="s-retention",
        now=now,
    )
    assert compacted is not None
    assert compacted.metadata["compaction_kind"] == "session_segment"
    timeline = manager.compact_timeline(
        tenant_id="tenant-a",
        owner_user_id="u1",
        timeline_path="timeline/2026/07/14",
        now=now,
    )
    assert timeline is not None
    assert timeline.record_kind == CatalogRecordKind.TREE_OVERVIEW.value
    persisted_evidence = archives.read_archive(evidence.archive_uri)
    assert persisted_evidence.archive_digest == evidence.archive_digest

    stale = _record(
        "stale",
        event_time=now - timedelta(days=2),
        lifecycle_state="deleted",
    )
    catalog.upsert_catalog(stale)
    vector.upsert_vector(stale.uri, [1.0, 0.0])
    stale_result = manager.gc_stale_projections(tenant_id="tenant-a")
    assert stale_result.stale_projections == 1
    assert catalog.get_catalog(stale.record_key) is None
    assert stale.uri not in vector.rows

    unsafe = _record("unsafe-tombstone", event_time=now - timedelta(days=2))
    catalog.upsert_catalog(unsafe)
    unsafe_tombstone = catalog.enqueue_tombstone(
        tenant_id="tenant-a",
        record_key=unsafe.record_key,
        uri=unsafe.uri,
        reason="source-delete-not-yet-proven",
        source_revision=unsafe.source_revision,
    )
    catalog.mark_tombstone_applied(str(unsafe_tombstone["tombstone_id"]))

    # Prove bounded path/tombstone GC on the real SQLite auxiliary tables.
    with sqlite3.connect(catalog.path) as conn:
        conn.execute(
            "INSERT INTO context_paths(tenant_id, record_key, uri, owner_user_id, context_type, "
            "event_time, path, path_kind, depth, is_primary, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "tenant-a",
                "missing-record",
                "memoryos://user/u1/missing",
                "u1",
                "session",
                now.isoformat(),
                "sessions/missing",
                "primary",
                2,
                1,
                now.isoformat(),
                now.isoformat(),
            ),
        )
    auxiliary = manager.gc_auxiliary_state(now=now + timedelta(days=61))
    assert auxiliary.orphan_paths_deleted == 1
    assert auxiliary.tombstones_deleted >= 1
    with sqlite3.connect(catalog.path) as conn:
        retained = conn.execute(
            "SELECT status FROM context_tombstones WHERE tombstone_id = ?",
            (str(unsafe_tombstone["tombstone_id"]),),
        ).fetchone()
    assert retained == ("APPLIED",)


def test_retention_policy_rejects_unsafe_thresholds(tmp_path) -> None:
    with pytest.raises(ValueError, match="monotonically"):
        RetentionPolicy(hot_for=timedelta(days=10), warm_for=timedelta(days=5))
    manager = CatalogRetentionManager(SQLiteIndexStore(tmp_path / "catalog.sqlite3"))
    with pytest.raises(ValueError, match="timezone"):
        manager.tier_for(
            _record("r", event_time=datetime.now(timezone.utc)),
            now=datetime.now(),
        )
