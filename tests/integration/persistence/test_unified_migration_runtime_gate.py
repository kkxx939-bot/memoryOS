from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import replace
from typing import Any, cast

import pytest

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.connect import ConnectMetadata
from memoryos.contextdb.catalog import CatalogRecord
from memoryos.contextdb.retrieval.orchestrator import RetrievalUnavailableError
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore
from memoryos.contextdb.unified_migration import MigrationState, ReadRoute
from memoryos.workers.session_commit_worker import SessionCommitWorker


def _commit_file_session(client: MemoryOSClient, session_id: str, file_name: str):  # noqa: ANN202
    return client.commit_agent_session(
        user_id="u1",
        session_id=session_id,
        messages=[{"role": "user", "content": "Read the desktop file."}],
        tool_results=[
            {
                "tool_name": "read_file",
                "output": f"contents of {file_name}",
                "path": f"/Users/u1/Desktop/{file_name}",
                "occurred_at": "2026-07-14T09:00:00+08:00",
            }
        ],
        connect_metadata=ConnectMetadata.default_agent("codex").to_dict(),
        project_id="migration-project",
        async_commit=False,
        tenant_id="tenant-a",
    )


def _archive(session_id: str, file_name: str) -> SessionArchive:
    return SessionArchive(
        user_id="u1",
        session_id=session_id,
        archive_uri=f"memoryos://user/u1/sessions/history/{session_id}",
        created_at="2026-07-14T09:00:00+08:00",
        metadata={
            "tenant_id": "tenant-a",
            "timezone": "Asia/Singapore",
            "connect": ConnectMetadata.default_agent("codex").to_dict(),
            "workspace_id": "migration-project",
        },
        messages=[
            {
                "role": "user",
                "content": "Read the desktop file.",
                "occurred_at": "2026-07-14T09:00:00+08:00",
            }
        ],
        tool_results=[
            {
                "tool_name": "read_file",
                "output": f"contents of {file_name}",
                "path": f"/Users/u1/Desktop/{file_name}",
                "occurred_at": "2026-07-14T09:00:00+08:00",
            }
        ],
    )


def _write_v2_catalog(path) -> None:  # noqa: ANN001
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE contexts (
              uri TEXT PRIMARY KEY,
              tenant_id TEXT NOT NULL,
              owner_user_id TEXT NOT NULL,
              context_type TEXT NOT NULL,
              project_id TEXT NOT NULL DEFAULT '',
              adapter_id TEXT NOT NULL DEFAULT '',
              admission_status TEXT NOT NULL DEFAULT '',
              claim_state TEXT NOT NULL DEFAULT '',
              slot_id TEXT NOT NULL DEFAULT '',
              memory_type TEXT NOT NULL DEFAULT '',
              scope_keys TEXT NOT NULL DEFAULT '[]',
              title TEXT NOT NULL,
              lifecycle_state TEXT NOT NULL,
              hotness REAL NOT NULL,
              semantic_hotness REAL NOT NULL,
              behavior_support_hotness REAL NOT NULL,
              metadata_json TEXT NOT NULL,
              content_text TEXT NOT NULL,
              updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO contexts VALUES (
              ?, 'tenant-a', 'u1', 'resource', 'migration-project', '', '', '', '', '', '[]',
              'Legacy report', 'active', 0, 0, 0, ?, ?, '2026-07-14T03:30:00+00:00'
            )
            """,
            (
                "memoryos://user/u1/resources/legacy-report",
                json.dumps({"summary": "legacy quarterly report"}),
                "legacy quarterly report",
            ),
        )
        conn.execute("CREATE TABLE contexts_fts(uri TEXT PRIMARY KEY, title, content_text, metadata_text)")
        conn.execute("PRAGMA user_version = 2")


def test_real_v2_first_start_binds_schema_ready_and_never_silently_skips_backfill(tmp_path) -> None:  # noqa: ANN001
    SessionArchiveStore(tmp_path, tenant_id="tenant-a").write_sync_archive(
        _archive("pre-upgrade-session", "archive-only.txt")
    )
    index_path = tmp_path / "tenants" / "tenant-a" / "indexes" / "context.sqlite3"
    _write_v2_catalog(index_path)

    client = MemoryOSClient(str(tmp_path), tenant_id="tenant-a")
    migration = client.unified_context_migration
    assert migration is not None
    state = client.index_store.get_migration_state(  # type: ignore[attr-defined]
        migration.migration_name,
        tenant_id="tenant-a",
    )
    assert state is not None
    assert state["state"] == MigrationState.SCHEMA_READY.value
    assert state["details_json"]["upgraded_from_schema_version"] == 2
    assert state["details_json"]["backfill_complete"] is False
    assert client.migration_gate is not None
    assert client.migration_gate.feature_gate.read_route is ReadRoute.LEGACY
    assert migration.initialize()["state"] == MigrationState.SCHEMA_READY.value

    legacy = client.assemble_context(
        "legacy quarterly report",
        user_id="u1",
        project_id="migration-project",
        tenant_id="tenant-a",
    )
    assert legacy["contexts"]
    assert "migration_legacy_compatible_read:SCHEMA_READY" in legacy["degraded_modes"]
    with pytest.raises(RetrievalUnavailableError, match="empty result cannot be proved"):
        client.search_context(
            "archive-only.txt",
            user_id="u1",
            project_id="migration-project",
            tenant_id="tenant-a",
        )

    migration.start_backfill()
    while migration.state is MigrationState.BACKFILLING:
        migration.backfill_next_batch()
    restored = client.search_context(
        "archive-only.txt",
        user_id="u1",
        project_id="migration-project",
        tenant_id="tenant-a",
    )
    assert any("archive-only.txt" in str(item.get("content") or "") for item in restored)

    restarted = MemoryOSClient(str(tmp_path), tenant_id="tenant-a")
    restarted_state = restarted.index_store.get_migration_state(  # type: ignore[attr-defined]
        migration.migration_name,
        tenant_id="tenant-a",
    )
    assert restarted_state is not None
    assert restarted_state["state"] == MigrationState.DUAL_WRITE.value


def test_greenfield_is_unified_but_existing_archive_with_fresh_catalog_requires_backfill(tmp_path) -> None:  # noqa: ANN001
    greenfield_root = tmp_path / "greenfield"
    greenfield = MemoryOSClient(str(greenfield_root), tenant_id="tenant-a")
    assert greenfield.migration_gate is not None
    assert greenfield.migration_gate.feature_gate.state is MigrationState.COMPLETED
    assert greenfield.migration_gate.feature_gate.read_route is ReadRoute.UNIFIED
    assert greenfield.index_store.get_migration_state(  # type: ignore[attr-defined]
        "unified-context-catalog-v1",
        tenant_id="tenant-a",
    ) is None
    assert greenfield.migration_gate.greenfield_catalog_origin_exists
    assert greenfield.search_context(
        "nothing yet",
        user_id="u1",
        project_id="migration-project",
        tenant_id="tenant-a",
    ) == []
    projected = _commit_file_session(greenfield, "greenfield-session", "greenfield.txt")
    assert projected.session_projection_status == "projected"
    greenfield_restarted = MemoryOSClient(str(greenfield_root), tenant_id="tenant-a")
    assert greenfield_restarted.migration_gate is not None
    assert greenfield_restarted.migration_gate.feature_gate.read_route is ReadRoute.UNIFIED
    assert any(
        "greenfield.txt" in str(item.get("content") or "")
        for item in greenfield_restarted.search_context(
            "greenfield.txt",
            user_id="u1",
            project_id="migration-project",
            tenant_id="tenant-a",
        )
    )

    archive_root = tmp_path / "archive-upgrade"
    SessionArchiveStore(archive_root, tenant_id="tenant-a").write_sync_archive(
        _archive("archive-before-catalog", "preexisting-archive.txt")
    )
    upgraded = MemoryOSClient(str(archive_root), tenant_id="tenant-a")
    migration = upgraded.unified_context_migration
    assert migration is not None
    state = upgraded.index_store.get_migration_state(  # type: ignore[attr-defined]
        migration.migration_name,
        tenant_id="tenant-a",
    )
    assert state is not None
    assert state["state"] == MigrationState.SCHEMA_READY.value
    assert state["details_json"]["backfill_reason"] == "existing_session_archive_evidence"
    assert state["details_json"]["backfill_complete"] is False
    assert upgraded.index_store.get_migration_state(  # type: ignore[attr-defined]
        migration.migration_name,
        tenant_id="default",
    ) is None
    with pytest.raises(RetrievalUnavailableError, match="empty result cannot be proved"):
        upgraded.search_context(
            "preexisting-archive.txt",
            user_id="u1",
            project_id="migration-project",
            tenant_id="tenant-a",
        )

    restarted = MemoryOSClient(str(archive_root), tenant_id="tenant-a")
    restarted_state = restarted.index_store.get_migration_state(  # type: ignore[attr-defined]
        migration.migration_name,
        tenant_id="tenant-a",
    )
    assert restarted_state is not None
    assert restarted_state["state"] == MigrationState.SCHEMA_READY.value


def test_cutover_fence_blocks_concurrent_archive_publish_and_forces_revalidation(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path), tenant_id="tenant-a")
    migration = client.unified_context_migration
    assert migration is not None
    migration.prepare_schema()
    migration.start_backfill()
    while migration.state is MigrationState.BACKFILLING:
        migration.backfill_next_batch()

    service = client.session_commit_service
    seed = _archive("fence-seed", "fence-seed.txt")
    service.sync_archive(seed, enqueue_commit_job=False)
    migration.minimum_shadow_samples = 1
    migration.start_shadow_validation()
    while not migration.validate_next_shadow_batch().complete:
        pass

    original_projector = service.session_projector
    projection_started = threading.Event()
    release_projection = threading.Event()

    class BlockingProjector:
        def project(self, archive: SessionArchive):  # noqa: ANN202
            projection_started.set()
            assert release_projection.wait(timeout=5)
            assert original_projector is not None
            return original_projector.project(archive)

    service.session_projector = BlockingProjector()
    failures: list[BaseException] = []

    def publish_late_archive() -> None:
        try:
            service.sync_archive(
                _archive("fence-late", "fence-late.txt"),
                enqueue_commit_job=False,
            )
        except BaseException as exc:  # pragma: no cover - asserted below.
            failures.append(exc)

    thread = threading.Thread(target=publish_late_archive)
    thread.start()
    assert projection_started.wait(timeout=5)
    with pytest.raises(TimeoutError, match="Lock already held"):
        migration.mark_ready_to_cutover()
    release_projection.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert failures == []
    service.session_projector = original_projector

    with pytest.raises(ValueError, match="source set changed"):
        migration.mark_ready_to_cutover()
    assert migration.state is MigrationState.SHADOW_VALIDATING


def test_runtime_gate_controls_dual_write_shadow_cutover_and_rollback(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path), tenant_id="tenant-a")
    index = cast(SQLiteIndexStore, client.index_store)
    migration = client.unified_context_migration
    assert migration is not None

    assert migration.initialize()["state"] == MigrationState.NOT_STARTED.value
    assert migration.prepare_schema()["state"] == MigrationState.SCHEMA_READY.value

    legacy_only = _commit_file_session(client, "before-cutover", "before-cutover.txt")
    assert legacy_only.session_projection_status == "migration_legacy_only"
    assert (
        index.list_catalog(
            filters={"tenant_id": "tenant-a", "session_ids": ("before-cutover",)},
            limit=20,
        )
        == []
    )
    with pytest.raises(RetrievalUnavailableError, match="empty result cannot be proved"):
        client.search_context(
            "before-cutover.txt",
            user_id="u1",
            project_id="migration-project",
            tenant_id="tenant-a",
        )

    migration.start_backfill()
    while migration.state is MigrationState.BACKFILLING:
        migration.backfill_next_batch()
    assert migration.state is MigrationState.DUAL_WRITE
    backfilled = client.search_context(
        "before-cutover.txt",
        user_id="u1",
        project_id="migration-project",
        tenant_id="tenant-a",
    )
    assert any("before-cutover.txt" in str(item.get("content") or "") for item in backfilled)

    dual_written = _commit_file_session(client, "dual-write", "dual-write.txt")
    assert dual_written.session_projection_status == "projected"

    migration.minimum_shadow_samples = 1
    migration.start_shadow_validation()
    for _ in range(10):
        if migration.validate_next_shadow_batch().complete:
            break
    else:  # pragma: no cover - bounded migration regression guard.
        raise AssertionError("shadow validation did not reach its bounded checkpoint")
    shadow = client.assemble_context(
        "dual-write.txt",
        user_id="u1",
        project_id="migration-project",
        tenant_id="tenant-a",
    )
    assert "migration_shadow_read:SHADOW_VALIDATING" in shadow["degraded_modes"]
    state = index.get_migration_state(
        migration.migration_name,
        tenant_id="tenant-a",
    )
    assert state is not None
    assert state["details_json"]["shadow_sample_count"] >= 1
    assert state["details_json"]["shadow_mismatch_count"] == 0
    assert state["details_json"]["shadow_read_sample_count"] >= 1
    assert state["details_json"]["shadow_read_mismatch_count"] == 0
    shadow_reads = index.list_migration_shadow_reads(
        migration.migration_name,
        tenant_id="tenant-a",
        validation_epoch=str(state["details_json"]["shadow_validation_epoch"]),
        limit=10,
    )
    assert len(shadow_reads) == 1 and shadow_reads[0]["matched"] == 1
    assert "dual-write.txt" not in str(shadow_reads)

    worker = SessionCommitWorker(client.session_commit_service)
    for _ in range(10):
        if client.queue_store.stats(queue_name="session_commit").get("pending", 0) == 0:
            break
        drained = worker.process_pending(batch_size=10)
        assert drained["failed"] == 0
    else:  # pragma: no cover - bounded cutover queue guard.
        raise AssertionError("session projection queue did not drain before cutover")
    migration.mark_ready_to_cutover()
    migration.cutover()
    assert migration.state is MigrationState.CUTOVER
    cutover = client.assemble_context(
        "dual-write.txt",
        user_id="u1",
        project_id="migration-project",
        tenant_id="tenant-a",
    )
    assert not any(str(mode).startswith("migration_") for mode in cutover["degraded_modes"])

    migration.rollback("operator requested safe rollback")
    assert migration.state is MigrationState.ROLLBACK
    # Damage adjuncts used exclusively by the Unified planner.  The rollback
    # read must remain available through the independent flat Catalog reader.
    with sqlite3.connect(index.path) as conn:
        keys = [
            str(row[0])
            for row in conn.execute(
                "SELECT record_key FROM contexts WHERE tenant_id = ? AND session_id = ?",
                ("tenant-a", "dual-write"),
            )
        ]
        for key in keys:
            conn.execute("DELETE FROM context_acl_grants WHERE record_key = ?", (key,))
            conn.execute("DELETE FROM context_path_acl WHERE record_key = ?", (key,))
            conn.execute("DELETE FROM context_path_closure WHERE record_key = ?", (key,))
    rollback_write = _commit_file_session(client, "rollback", "rollback-only.txt")
    assert rollback_write.session_projection_status == "projected"
    rollback_read = client.assemble_context(
        "dual-write.txt",
        user_id="u1",
        project_id="migration-project",
        tenant_id="tenant-a",
    )
    assert rollback_read["contexts"]
    assert "migration_legacy_compatible_read:ROLLBACK" in rollback_read["degraded_modes"]
    assert index.list_catalog(
        filters={"tenant_id": "tenant-a", "session_ids": ("rollback",)},
        limit=20,
    )
    rollback_only = client.search_context(
        "rollback-only.txt",
        user_id="u1",
        project_id="migration-project",
        tenant_id="tenant-a",
    )
    assert any("rollback-only.txt" in str(item.get("content") or "") for item in rollback_only)
    archive_compat = client.archive_search(
        "rollback-only.txt",
        user_id="u1",
        project_id="migration-project",
        tenant_id="tenant-a",
    )
    assert any("rollback-only.txt" in str(item.get("preview") or "") for item in archive_compat)


def test_runtime_migration_backfills_existing_current_slot_before_cutover(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path), tenant_id="tenant-a")
    remembered = client.remember(
        user_id="u1",
        memory_type="preference",
        content="I like ice cream",
        identity_fields={"subject": "food", "dimension": "ice_cream"},
        tenant_id="tenant-a",
    )
    assert remembered["status"] == "COMMITTED"
    current_rows = [
        record
        for record in client.index_store.list_catalog(  # type: ignore[attr-defined]
            filters={"tenant_id": "tenant-a", "record_kinds": ("current_slot",)},
            limit=20,
        )
        if isinstance(record, CatalogRecord)
    ]
    assert len(current_rows) == 1

    # Simulate an upgraded installation whose authoritative Slot/Claim heads
    # predate the v6 serving catalog.  Clearing derived state never touches the
    # canonical Source, receipt, current head, or evidence.
    client.index_store.clear()
    assert not client.index_store.list_catalog(  # type: ignore[attr-defined]
        filters={"tenant_id": "tenant-a", "record_kinds": ("current_slot",)},
        limit=20,
    )

    migration = client.unified_context_migration
    assert migration is not None
    migration.initialize()
    migration.prepare_schema()
    migration.start_backfill()
    during_backfill = client.remember(
        user_id="u1",
        memory_type="preference",
        content="I like sorbet",
        identity_fields={"subject": "food", "dimension": "sorbet"},
        tenant_id="tenant-a",
    )
    assert during_backfill["status"] == "COMMITTED"
    during_rows = client.index_store.list_catalog(  # type: ignore[attr-defined]
        filters={
            "tenant_id": "tenant-a",
            "record_kinds": ("current_slot",),
        },
        limit=20,
    )
    assert len(during_rows) == 1
    assert "sorbet" in str(during_rows[0].metadata["canonical_value"])
    while migration.state is MigrationState.BACKFILLING:
        migration.backfill_next_batch()
    restored = client.index_store.list_catalog(  # type: ignore[attr-defined]
        filters={"tenant_id": "tenant-a", "record_kinds": ("current_slot",)},
        limit=20,
    )
    assert len(restored) == 2
    restored_old = next(row for row in restored if row.canonical_slot_id == current_rows[0].canonical_slot_id)
    assert restored_old.canonical_claim_id == current_rows[0].canonical_claim_id
    state = client.index_store.get_migration_state(  # type: ignore[attr-defined]
        migration.migration_name,
        tenant_id="tenant-a",
    )
    assert state is not None
    assert state["details_json"]["canonical_backfill_complete"] is True
    assert state["details_json"]["backfilled_canonical_slots"] == 2
    proofs = client.index_store.list_migration_equivalence_proofs(  # type: ignore[attr-defined]
        migration.migration_name,
        tenant_id="tenant-a",
        limit=20,
    )
    assert "canonical_current_slot" in {proof["plane"] for proof in proofs}


def test_backfill_inline_projection_failure_enqueues_durable_replay_after_checkpoint(tmp_path) -> None:
    client = MemoryOSClient(str(tmp_path), tenant_id="tenant-a")
    service = client.session_commit_service
    archive_store = service.archive_store
    archive_store.write_sync_archive(_archive("s2", "s2.txt"))
    archive_store.write_sync_archive(_archive("s3", "s3.txt"))
    migration = client.unified_context_migration
    assert migration is not None
    migration.batch_size = 1
    migration.prepare_schema()
    migration.start_backfill()
    first = migration.backfill_next_batch()
    assert first.state is MigrationState.BACKFILLING
    assert "s2" in first.checkpoint

    projector = service.session_projector

    class BrokenProjector:
        def project(self, archive: SessionArchive) -> object:
            raise RuntimeError(f"injected projection failure for {archive.session_id}")

    service.session_projector = BrokenProjector()
    earlier = _archive("s1", "earlier.txt")
    with pytest.raises(RuntimeError, match="injected projection failure"):
        client.context_db.commit_session(earlier, async_commit=True)
    assert archive_store.archive_exists(earlier.archive_uri, tenant_id="tenant-a")
    assert client.queue_store.stats(queue_name="session_commit").get("pending", 0) == 1

    service.session_projector = projector
    replay = SessionCommitWorker(service).process_pending(batch_size=1)
    assert replay["claimed"] == 1
    assert client.index_store.list_catalog(  # type: ignore[attr-defined]
        filters={"tenant_id": "tenant-a", "session_ids": ("s1",)},
        limit=100,
    )


def test_late_shadow_projection_failure_restarts_epoch_and_blocks_cutover(tmp_path) -> None:
    client = MemoryOSClient(str(tmp_path), tenant_id="tenant-a")
    service = client.session_commit_service
    migration = client.unified_context_migration
    assert migration is not None
    client.context_db.commit_session(_archive("shadow-seed", "seed.txt"), async_commit=True)
    migration.prepare_schema()
    migration.start_backfill()
    while migration.state is MigrationState.BACKFILLING:
        migration.backfill_next_batch()
    migration.minimum_shadow_samples = 1
    migration.start_shadow_validation()
    while not migration.validate_next_shadow_batch().complete:
        pass
    original_projector = service.session_projector

    class BrokenProjector:
        def project(self, archive: SessionArchive) -> object:
            raise RuntimeError(f"injected late projection failure for {archive.session_id}")

    service.session_projector = BrokenProjector()
    late = _archive("a-late-shadow", "late-shadow.txt")
    with pytest.raises(RuntimeError, match="injected late projection failure"):
        client.context_db.commit_session(late, async_commit=False)
    assert service.archive_store.archive_exists(late.archive_uri, tenant_id="tenant-a")
    assert client.queue_store.stats(queue_name="session_commit").get("pending", 0) == 1
    assert (
        client.index_store.list_catalog(  # type: ignore[attr-defined]
            filters={"tenant_id": "tenant-a", "session_ids": (late.session_id,)},
            limit=20,
        )
        == []
    )

    with pytest.raises(ValueError, match="source set changed"):
        migration.mark_ready_to_cutover()
    assert migration.state is MigrationState.SHADOW_VALIDATING

    service.session_projector = original_projector
    replay = SessionCommitWorker(service).process_pending(batch_size=10)
    assert replay["failed"] == 0
    assert client.queue_store.stats(queue_name="session_commit").get("pending", 0) == 0
    while not migration.validate_next_shadow_batch().complete:
        pass
    shadow_read = client.assemble_context(
        "late-shadow.txt",
        user_id="u1",
        project_id="migration-project",
        tenant_id="tenant-a",
    )
    assert "migration_shadow_read:SHADOW_VALIDATING" in shadow_read["degraded_modes"]
    assert migration.mark_ready_to_cutover()["state"] == MigrationState.READY_TO_CUTOVER.value
    assert migration.cutover()["state"] == MigrationState.CUTOVER.value
    assert client.index_store.list_catalog(  # type: ignore[attr-defined]
        filters={"tenant_id": "tenant-a", "session_ids": (late.session_id,)},
        limit=20,
    )


@pytest.mark.parametrize("damage", ["missing", "tampered"])
def test_shadow_current_slot_validation_detects_damage_without_repairing_it(tmp_path, damage: str) -> None:
    client = MemoryOSClient(str(tmp_path), tenant_id="tenant-a")
    remembered = client.remember(
        user_id="u1",
        memory_type="preference",
        content="I like gelato",
        identity_fields={"subject": "food", "dimension": "gelato"},
        tenant_id="tenant-a",
    )
    assert remembered["status"] == "COMMITTED"
    migration = client.unified_context_migration
    assert migration is not None
    migration.prepare_schema()
    migration.start_backfill()
    while migration.state is MigrationState.BACKFILLING:
        migration.backfill_next_batch()
    rows = client.index_store.list_catalog(  # type: ignore[attr-defined]
        filters={"tenant_id": "tenant-a", "record_kinds": ("current_slot",)},
        limit=20,
    )
    assert len(rows) == 1
    canonical_prover = getattr(migration.canonical_current_backfill, "prove", None)
    assert callable(canonical_prover)
    intact: Any = canonical_prover("", 20)
    assert len(intact.equivalence_proofs) == 1
    assert intact.equivalence_proofs[0].matched
    damaged_key = rows[0].record_key
    if damage == "missing":
        assert client.index_store.delete_catalog(damaged_key, tenant_id="tenant-a")  # type: ignore[attr-defined]
    else:
        client.index_store.upsert_catalog(replace(rows[0], l0_text="tampered derived current"))  # type: ignore[attr-defined]

    migration.minimum_shadow_samples = 1
    migration.start_shadow_validation()
    validation = migration.validate_next_shadow_batch()
    assert validation.complete
    assert validation.processed_canonical_slots == 1
    assert validation.mismatch_count == 1
    damaged = client.index_store.get_catalog(damaged_key, tenant_id="tenant-a")  # type: ignore[attr-defined]
    if damage == "missing":
        assert damaged is None
    else:
        assert damaged is not None and damaged.l0_text == "tampered derived current"
    with pytest.raises(ValueError, match="mismatch threshold"):
        migration.mark_ready_to_cutover()
