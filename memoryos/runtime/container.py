"""运行时里的依赖组装。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.retention import CatalogRetentionManager, RetentionPolicy
from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
from memoryos.contextdb.session.commit_group import CommitGroupStore
from memoryos.contextdb.session.context_projector import SessionContextProjector
from memoryos.contextdb.session.planners import ActionPolicyCommitPlanner, BehaviorCommitPlanner, MemoryCommitPlanner
from memoryos.contextdb.session.planning_envelope import PlanningEnvelopeStore
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.session.session_commit import SessionCommitService
from memoryos.contextdb.store import FileSystemSourceStore, IndexStore, RelationStore, SourceStore
from memoryos.contextdb.store.source_store import LockStore, QueueStore
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore
from memoryos.contextdb.store.sqlite_lock_store import SQLiteLockStore
from memoryos.contextdb.store.sqlite_queue_store import SQLiteQueueStore
from memoryos.contextdb.store.sqlite_relation_store import SQLiteRelationStore
from memoryos.contextdb.store.vector_store import (
    VectorStore,
    require_production_vector_capabilities,
    vector_capabilities,
)
from memoryos.contextdb.tombstone import ProjectionTombstoneService
from memoryos.contextdb.transaction.recovery import RecoveryService
from memoryos.contextdb.unified_migration import (
    CurrentSlotMigrationBackfill,
    RuntimeMigrationCoordinator,
    UnifiedContextMigration,
    has_existing_session_archive_evidence,
)
from memoryos.core.path_safety import validate_authoritative_tree
from memoryos.memory.canonical.event import canonical_digest
from memoryos.memory.canonical.history import validate_canonical_receipt_history
from memoryos.memory.canonical.identity import AliasRegistry
from memoryos.memory.canonical.migration import MemoryClosureMigration
from memoryos.memory.canonical.projection import CanonicalMemoryProjector, MemoryProjectionWorker
from memoryos.memory.canonical.projection_state import ProjectionIntegrityError, ProjectionRecordStore
from memoryos.memory.canonical.repository import CanonicalMemoryRepository
from memoryos.memory.canonical.review_command import validate_pending_review_commands
from memoryos.memory.canonical.salience_ledger import DurableSalienceLedger
from memoryos.memory.canonical.slot_projection import CurrentSlotProjection
from memoryos.memory.canonical.visibility import reconcile_committed_relation_store
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.commit.planning_proof import ImmutablePlanningProofStore
from memoryos.prediction.model.prediction_ledger import PredictionLedger
from memoryos.prediction.pipeline.executor import ActionExecutor
from memoryos.prediction.pipeline.prediction_engine import PredictionEngine
from memoryos.providers.embedding import EmbeddingProvider
from memoryos.providers.rerank import Reranker
from memoryos.runtime.config import RuntimeConfig
from memoryos.runtime.readiness import RuntimeReadiness, RuntimeReadinessState
from memoryos.skill.tool_registry import ToolRegistry
from memoryos.workers.recovery_worker import RecoveryWorker


@dataclass
class RuntimeContainer:
    """把 SDK、接口和后台任务共用的运行组件放在一起。"""

    source_store: SourceStore
    index_store: IndexStore
    relation_store: RelationStore
    queue_store: QueueStore
    lock_store: LockStore
    vector_store: VectorStore | None
    embedding_provider: EmbeddingProvider | None
    hybrid_search: HybridSearch | None
    reranker: Reranker | None
    committer: OperationCommitter
    session_archive_store: SessionArchiveStore
    session_commit_service: SessionCommitService
    context_db: ContextDB
    engine: PredictionEngine
    executor: ActionExecutor
    memory_projection_worker: MemoryProjectionWorker
    recovery_service: RecoveryService
    recovery_worker: RecoveryWorker
    readiness: RuntimeReadiness
    tombstone_service: ProjectionTombstoneService | None = None
    retention_manager: CatalogRetentionManager | None = None
    migration_gate: RuntimeMigrationCoordinator | None = None
    unified_context_migration: UnifiedContextMigration | None = None


@contextmanager
def _runtime_projection_fence(
    migration_gate: RuntimeMigrationCoordinator | None,
) -> Iterator[None]:
    """Fence startup derived repair against another runtime's rebuild."""

    acquire = getattr(migration_gate, "acquire_projection_fence", None)
    release = getattr(migration_gate, "release_projection_fence", None)
    fence = acquire() if callable(acquire) else None
    try:
        yield
    finally:
        if callable(release):
            release(fence)


def build_runtime_container(
    config: RuntimeConfig,
    *,
    index_store: IndexStore | None = None,
    source_store: SourceStore | None = None,
    relation_store: RelationStore | None = None,
    queue_store: QueueStore | None = None,
    lock_store: LockStore | None = None,
    tool_registry: ToolRegistry | None = None,
    vector_store: VectorStore | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    hybrid_search: HybridSearch | None = None,
) -> RuntimeContainer:
    """组装默认运行链路，并拒绝会直接生成数据库操作的旧提取器。"""

    if config.memory_extractor is not None and (
        not getattr(config.memory_extractor, "semantic_proposal_backend", False)
        or not getattr(config.memory_extractor, "llm_semantic_backend", False)
    ):
        raise TypeError("memory_extractor must be an LLM MemorySemanticProposal backend")
    root_path = config.root_path
    readiness = RuntimeReadiness()
    root_path.mkdir(parents=True, exist_ok=True, mode=0o700)
    root_path.chmod(0o700)
    source = source_store or FileSystemSourceStore(root_path, tenant_id=config.tenant_id)
    if hasattr(source, "__dict__"):
        vars(source)["readiness"] = readiness
    source_tenant = getattr(source, "tenant_id", config.tenant_id)
    if str(source_tenant) != config.tenant_id:
        raise ValueError("SourceStore tenant does not match RuntimeConfig tenant_id")
    tenant_root = root_path if config.tenant_id == "default" else root_path / "tenants" / config.tenant_id
    tenant_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    tenant_root.chmod(0o700)
    index_root = tenant_root / "indexes"
    index = index_store or SQLiteIndexStore(index_root / "context.sqlite3")
    relation = relation_store or SQLiteRelationStore(index_root / "relations.sqlite3")
    queue = queue_store or SQLiteQueueStore(tenant_root / "queues" / "jobs.sqlite3")
    lock = lock_store or SQLiteLockStore(root_path / "system" / "locks.sqlite3")
    session_archive_store = SessionArchiveStore(root_path, tenant_id=config.tenant_id)
    has_preexisting_session_evidence = has_existing_session_archive_evidence(
        session_archive_store,
        tenant_id=config.tenant_id,
    )
    migration_gate = RuntimeMigrationCoordinator(
        cast(Any, index),
        tenant_id=config.tenant_id,
        lock_store=lock,
    )
    if hasattr(source, "__dict__"):
        vars(source)["migration_gate"] = migration_gate
    schema_version_reader = getattr(index, "catalog_schema_version", None)
    if has_preexisting_session_evidence and not migration_gate.greenfield_catalog_origin_exists:
        if not callable(schema_version_reader):
            raise RuntimeError("existing SessionArchive evidence requires a versioned Catalog migration store")
        raw_schema_version: Any = schema_version_reader()
        if not isinstance(raw_schema_version, int):
            raise RuntimeError("existing SessionArchive evidence requires a valid Catalog schema version")
        migration_gate.require_backfill(
            reason="existing_session_archive_evidence",
            schema_version=raw_schema_version,
        )
    elif not has_preexisting_session_evidence:
        migration_gate.record_greenfield_catalog_origin()
    configured_embedding = embedding_provider or config.embedding
    configured_vector_store = vector_store or config.vector_store
    configured_vector_capabilities = vector_capabilities(configured_vector_store)
    if config.mode == "server" and configured_vector_store is not None:
        configured_vector_capabilities = require_production_vector_capabilities(configured_vector_store)
    search = hybrid_search or (
        HybridSearch(
            index, vector_store=configured_vector_store, embedding_provider=configured_embedding, source_store=source
        )
        if configured_vector_store is not None and configured_embedding is not None
        else None
    )
    tombstone_service = (
        ProjectionTombstoneService(
            index,
            source_store=source,
            vector_store=configured_vector_store,
            relation_store=relation,
        )
        if callable(getattr(index, "enqueue_tombstone", None))
        else None
    )
    aliases = AliasRegistry(config.memory_aliases)
    committer = OperationCommitter(
        source,
        index,
        config.root,
        lock_store=lock,
        relation_store=relation,
        queue_store=queue,
        tenant_id=config.tenant_id,
        alias_registry=aliases,
        tombstone_service=tombstone_service,
        migration_gate=migration_gate,
    )
    session_projector = (
        SessionContextProjector(
            cast(Any, index),
            vector_store=configured_vector_store,
            embedding_provider=configured_embedding,
            vectorize_important_events=bool(
                dict(config.retrieval or {}).get("vectorize_important_session_events", False)
            ),
        )
        if callable(getattr(index, "upsert_catalog", None))
        else None
    )
    projection_store = ProjectionRecordStore(tenant_root)
    canonical_projector = CanonicalMemoryProjector(
        source,
        index,
        tenant_root,
        relation_store=relation,
        vector_store=configured_vector_store,
        embedding_provider=configured_embedding,
        record_store=projection_store,
    )
    current_slot_projector = (
        CurrentSlotProjection(
            CanonicalMemoryRepository(source, relation),
            cast(Any, index),
            vector_store=configured_vector_store,
            embedding_provider=configured_embedding,
        )
        if callable(getattr(index, "upsert_catalog", None)) and callable(getattr(index, "apply_tombstone", None))
        else None
    )
    memory_projection_worker = MemoryProjectionWorker(
        canonical_projector,
        queue,
        current_slot_projector=current_slot_projector,
        migration_gate=migration_gate,
    )
    retention_manager = (
        CatalogRetentionManager(
            cast(Any, index),
            vector_store=configured_vector_store,
            tombstone_service=tombstone_service,
            policy=RetentionPolicy.from_config(config.retention),
        )
        if all(
            callable(getattr(index, name, None))
            for name in (
                "scan_catalog_batch",
                "enqueue_tombstone",
                "gc_orphan_paths",
                "gc_applied_tombstones",
            )
        )
        else None
    )
    recovery_service = RecoveryService(committer.redo, committer)
    recovery_worker = RecoveryWorker(recovery_service)
    session_commit_service = SessionCommitService(
        session_archive_store,
        queue,
        committer=committer,
        memory_planner=MemoryCommitPlanner(
            source_store=source,
            index_store=index,
            relation_store=relation,
            hybrid_search=search,
            extractor=config.memory_extractor,
            egress_policy=config.memory_egress_policy,
            alias_registry=aliases,
        ),
        behavior_planner=BehaviorCommitPlanner(index_store=index, source_store=source),
        action_policy_planner=ActionPolicyCommitPlanner(index_store=index, source_store=source),
        projection_worker=memory_projection_worker,
        session_projector=session_projector,
        migration_gate=migration_gate,
        commit_group_store=CommitGroupStore(tenant_root),
    )
    unified_context_migration = (
        UnifiedContextMigration(
            cast(Any, index),
            session_archive_store,
            session_projector,
            tenant_id=config.tenant_id,
            queue_store=queue,
            lock_store=lock,
            canonical_current_backfill=(
                CurrentSlotMigrationBackfill(source, current_slot_projector)
                if current_slot_projector is not None
                else None
            ),
        )
        if session_projector is not None
        and callable(getattr(index, "get_migration_state", None))
        and callable(getattr(index, "set_migration_state", None))
        else None
    )
    context_db = ContextDB(
        source,
        index,
        relation,
        queue_store=queue,
        session_commit_service=session_commit_service,
        committer=committer,
        projection_store=projection_store,
        canonical_projector=canonical_projector,
        current_slot_projector=current_slot_projector,
        tombstone_service=tombstone_service,
        retention_manager=retention_manager,
        migration_gate=migration_gate,
        unified_context_migration=unified_context_migration,
        readiness=readiness,
    )
    engine = PredictionEngine(
        index,
        PredictionLedger(config.root),
        source_store=source,
        relation_store=relation,
        vector_store=configured_vector_store,
        embedding_provider=configured_embedding,
        hybrid_search=search,
    )
    readiness.transition(RuntimeReadinessState.RECOVERING)
    startup_details: dict[str, Any] = {}
    try:
        startup_details["vector_capabilities"] = {
            "configured": configured_vector_store is not None,
            "supports_metadata_filtering": configured_vector_capabilities.supports_metadata_filtering,
            "supports_namespace_filtering": configured_vector_capabilities.supports_namespace_filtering,
            "supports_time_filtering": configured_vector_capabilities.supports_time_filtering,
            "supports_delete_by_filter": configured_vector_capabilities.supports_delete_by_filter,
            "production_filtered_top_k_ready": (configured_vector_capabilities.production_filtered_top_k_ready),
        }
        startup_details["artifact_path_topology"] = {
            "system": validate_authoritative_tree(
                tenant_root / "system",
                label="canonical system artifact tree",
            ),
            "views": validate_authoritative_tree(
                tenant_root / "views",
                label="canonical projection view tree",
            ),
        }
        migration = MemoryClosureMigration(
            root_path,
            tenant_id=config.tenant_id,
            source_store=source,
            relation_store=relation,
        )
        with _runtime_projection_fence(migration_gate):
            startup_details["migration"] = migration.run(allow_inflight=True)
        planning_envelopes = PlanningEnvelopeStore(
            root_path,
            tenant_id=config.tenant_id,
        )
        salience_ledger = DurableSalienceLedger(root_path, tenant_id=config.tenant_id)
        startup_details["salience_reservations"] = salience_ledger.validate_all()
        startup_details["planning_envelopes"] = planning_envelopes.validate_all()
        startup_details["planning_salience_links"] = _validate_planning_salience_links(
            planning_envelopes,
            salience_ledger,
        )
        startup_details["commit_group_planning_links"] = _validate_commit_group_planning_links(
            session_commit_service,
            planning_envelopes,
            salience_ledger,
        )
        startup_details["planning_proofs"] = ImmutablePlanningProofStore(
            tenant_root,
            tenant_id=config.tenant_id,
        ).validate_all()
        startup_details["projection_proof_structure"] = memory_projection_worker.proof_store.validate_all()
        recovery_result = recovery_worker.process_all()
        startup_details["recovery"] = recovery_result
        if recovery_result.get("failed_count") or recovery_result.get("quarantine_count"):
            failure = str(recovery_result.get("last_error") or "RecoveryIncomplete")
            raise RuntimeError(f"startup recovery left failed or quarantined transaction artifacts: {failure}")
        startup_details["projection_tombstones"] = _replay_startup_projection_tombstones(
            tombstone_service,
            index,
            migration_gate=migration_gate,
        )
        startup_details["migration_post_recovery"] = migration.validate_current_state()
        # Establish an authoritative committed baseline before resuming a
        # semantic commit group.  In particular, a corrupt historical receipt
        # that is no longer current must not be discovered only after startup
        # has already advanced another archive, and an unexpired queue lease
        # owned by another worker must not race startup recovery.
        startup_details["receipt_history_pre_commit_groups"] = validate_canonical_receipt_history(
            tenant_root,
            tenant_id=config.tenant_id,
        )
        startup_details["canonical_domains_pre_commit_groups"] = _validate_startup_canonical_domains(
            source,
            relation,
        )
        _validate_startup_heads(source, relation)
        # Recover and fence only queues that can participate in this memory
        # startup.  Other subsystems share the physical QueueStore, so their
        # terminal or live work is not evidence about canonical readiness.
        # Session/proposal dead letters are explicit terminal archive outcomes;
        # only projection terminal work means a committed canonical effect is
        # missing its required derived publication.
        startup_queue_names = ("session_commit", "memory_proposal", "memory_projection")
        with _runtime_projection_fence(migration_gate):
            recovered_by_queue = {
                queue_name: queue.recover_expired_leases(queue_name=queue_name)
                for queue_name in startup_queue_names
            }
        startup_details["queue_lease_recovery"] = {
            "recovered_expired": sum(recovered_by_queue.values()),
        }
        startup_details["queue_lease_recovery_by_queue"] = recovered_by_queue
        startup_details["session_projection_frontier_recovery"] = (
            session_commit_service.recover_session_projection_frontier()
        )
        queue_recovery_preflight = {
            queue_name: queue.stats(queue_name=queue_name) for queue_name in startup_queue_names
        }
        startup_details["queue_recovery_preflight"] = queue_recovery_preflight
        for queue_name, queue_state in queue_recovery_preflight.items():
            if queue_state.get("quarantine", 0):
                raise RuntimeError(f"startup {queue_name} queue contains quarantined work")
            if queue_state.get("leased", 0):
                raise RuntimeError(f"startup {queue_name} queue contains an active lease")
        projection_pre_recovery = queue_recovery_preflight["memory_projection"]
        if projection_pre_recovery.get("dead_letter", 0):
            raise RuntimeError("startup memory_projection queue contains dead-letter work")
        startup_details["commit_groups"] = _recover_startup_commit_groups(session_commit_service)
        startup_details["commit_group_effect_links"] = _validate_commit_group_effect_links(
            session_commit_service,
            committer,
        )
        startup_details["receipt_history"] = validate_canonical_receipt_history(
            tenant_root,
            tenant_id=config.tenant_id,
        )
        startup_details["canonical_domains"] = _validate_startup_canonical_domains(
            source,
            relation,
        )
        startup_details["pending_review_commands"] = validate_pending_review_commands(
            root_path,
            tenant_id=config.tenant_id,
            source_store=source,
            relation_store=relation,
        )
        _validate_startup_heads(source, relation)
        with _runtime_projection_fence(migration_gate):
            startup_details["canonical_relations"] = reconcile_committed_relation_store(
                source,
                relation,
            )
        # The durable rebuild gate may survive a process crash after Catalog
        # clear.  Resume only after transaction/queue/head recovery has made
        # all immutable inputs quiescent, and before READY can expose reads.
        startup_details["derived_serving_rebuild_recovery"] = (
            context_db.resume_derived_serving_rebuild_if_needed()
        )
        # Outbox, queue identity and immutable projection proofs are the
        # authoritative inputs to every projection rebuild.  Validate them
        # before touching index/vector/view state: a corrupt queue member or
        # a live worker lease must leave the existing derived snapshot intact.
        startup_details["projection_dispatch"] = memory_projection_worker.dispatch_outbox()
        memory_projection_worker._validate_authoritative_projection_proofs()
        startup_details["projection_authoritative_preflight"] = {"validated": True}
        pre_projection_queue_stats = queue.stats(queue_name="memory_projection")
        startup_details["projection_queue_preflight"] = pre_projection_queue_stats
        if pre_projection_queue_stats.get("dead_letter", 0) or pre_projection_queue_stats.get("quarantine", 0):
            raise RuntimeError("startup memory_projection queue contains dead-letter or quarantined work")
        if pre_projection_queue_stats.get("leased", 0):
            raise RuntimeError("startup memory_projection queue contains an active lease")
        projection_repairs: list[str] = []
        for repair_attempt in range(3):
            try:
                # Projection records, current pointers and index/vector/views
                # are rebuildable.  Repair them before consuming completion
                # jobs so a corrupt derived current cannot dead-letter an
                # otherwise intact immutable transaction/outbox chain.
                with _runtime_projection_fence(migration_gate):
                    startup_details["projection_prequeue"] = canonical_projector.rebuild(clear_views=True)
                    startup_details["projection_prequeue_validation"] = (
                        memory_projection_worker.verify_current_projections()
                    )
                break
            except ProjectionIntegrityError as exc:
                projection_repairs.append(f"{type(exc).__name__}: {exc}")
                if repair_attempt == 2:
                    raise
        projection_runs: list[dict[str, Any]] = []
        for _ in range(100):
            run = memory_projection_worker._process_pending_during_startup(
                limit=10,
                lease_seconds=300,
            )
            projection_runs.append(run)
            if not run["processed"] and not run["failed"]:
                break
        startup_details["projection_queue"] = projection_runs
        for repair_attempt in range(3):
            try:
                # Scope/taxonomy currents are disposable.  Always rebuild the
                # complete publication set so views for claims with no current
                # head cannot survive an otherwise clean startup.
                with _runtime_projection_fence(migration_gate):
                    startup_details["projection"] = canonical_projector.rebuild(clear_views=True)
                    startup_details["projection_validation"] = (
                        memory_projection_worker.verify_current_projections()
                    )
                    startup_details["projection_commit_groups"] = _validate_completed_projection_consumers(
                        session_commit_service,
                        memory_projection_worker,
                    )
                    startup_details["projection_proofs"] = memory_projection_worker.validate_projection_proofs()
                break
            except ProjectionIntegrityError as exc:
                projection_repairs.append(f"{type(exc).__name__}: {exc}")
                if repair_attempt == 2:
                    raise
        if current_slot_projector is not None and migration_gate.feature_gate.dual_write_enabled:
            with _runtime_projection_fence(migration_gate):
                current_backfill = CurrentSlotMigrationBackfill(source, current_slot_projector)
                current_checkpoint = ""
                current_slots = 0
                current_records = 0
                while True:
                    current_batch = current_backfill(current_checkpoint, 256)
                    current_slots += current_batch.processed_slots
                    current_records += current_batch.projected_records
                    current_checkpoint = current_batch.checkpoint
                    if current_batch.complete:
                        break
            startup_details["current_slot_projection_rebuild"] = {
                "processed_slots": current_slots,
                "projected_records": current_records,
                "checkpoint": current_checkpoint,
                "complete": True,
            }
        else:
            startup_details["current_slot_projection_rebuild"] = {
                "processed_slots": 0,
                "projected_records": 0,
                "complete": False,
                "reason": "migration_dual_write_disabled",
            }
        startup_details["projection_repairs"] = projection_repairs
        projection_queue_stats = queue.stats(queue_name="memory_projection")
        startup_details["projection_queue_final"] = projection_queue_stats
        if projection_queue_stats.get("dead_letter", 0) or projection_queue_stats.get("quarantine", 0):
            raise RuntimeError("startup memory_projection queue contains dead-letter or quarantined work")
        if projection_queue_stats.get("leased", 0):
            raise RuntimeError("startup memory_projection queue contains an active lease")
        if projection_queue_stats.get("pending", 0):
            raise RuntimeError("startup memory_projection queue still contains pending work")
        # Preserve all-queue telemetry without using unrelated terminal work as
        # a canonical projection completion proof.
        startup_details["queue"] = queue.stats()
        _validate_startup_heads(source, relation)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        readiness.transition(
            RuntimeReadinessState.NOT_READY,
            reasons=(f"{type(exc).__name__}: {exc}",),
            details=startup_details,
        )
    except Exception as exc:
        # Startup is the serving safety boundary.  An unclassified component
        # failure must remain observable while still returning a runtime whose
        # public memory APIs are gated by an explicit NOT_READY state.  Process
        # control exceptions derive from BaseException and deliberately escape.
        readiness.transition(
            RuntimeReadinessState.NOT_READY,
            reasons=(f"{type(exc).__name__}: {exc}",),
            details=startup_details,
        )
    else:
        readiness.transition(RuntimeReadinessState.READY, details=startup_details)
    return RuntimeContainer(
        source_store=source,
        index_store=index,
        relation_store=relation,
        queue_store=queue,
        lock_store=lock,
        vector_store=configured_vector_store,
        embedding_provider=configured_embedding,
        hybrid_search=search,
        reranker=config.reranker,
        committer=committer,
        session_archive_store=session_archive_store,
        session_commit_service=session_commit_service,
        context_db=context_db,
        engine=engine,
        executor=ActionExecutor(tool_registry),
        memory_projection_worker=memory_projection_worker,
        recovery_service=recovery_service,
        recovery_worker=recovery_worker,
        readiness=readiness,
        tombstone_service=tombstone_service,
        retention_manager=retention_manager,
        migration_gate=migration_gate,
        unified_context_migration=unified_context_migration,
    )


def _replay_startup_projection_tombstones(
    service: ProjectionTombstoneService | None,
    index_store: object,
    *,
    migration_gate: RuntimeMigrationCoordinator | None = None,
) -> dict[str, int]:
    """Drain durable DELETE outboxes before the runtime becomes readable."""

    acquire = getattr(migration_gate, "acquire_projection_fence", None)
    release = getattr(migration_gate, "release_projection_fence", None)
    fence = acquire() if callable(acquire) else None
    try:
        return _replay_startup_projection_tombstones_unfenced(service, index_store)
    finally:
        if callable(release):
            release(fence)


def _replay_startup_projection_tombstones_unfenced(
    service: ProjectionTombstoneService | None,
    index_store: object,
) -> dict[str, int]:
    """Implementation for a caller already holding the tenant fence."""

    if service is None:
        return {"processed": 0, "stale": 0, "batches": 0}
    pending = getattr(index_store, "get_pending_tombstones", None)
    if not callable(pending):
        raise RuntimeError("production Catalog has no durable tombstone replay API")
    processed = 0
    stale = 0
    batches = 0
    while True:
        rows = pending(limit=1_000)
        if not isinstance(rows, list):
            raise RuntimeError("durable tombstone replay returned an invalid batch")
        if not rows:
            return {"processed": processed, "stale": stale, "batches": batches}
        tombstone_ids = tuple(str(row.get("tombstone_id") or "") for row in rows if isinstance(row, dict))
        if len(tombstone_ids) != len(rows) or any(not tombstone_id for tombstone_id in tombstone_ids):
            raise RuntimeError("durable tombstone replay returned an invalid identity")
        result = service.process_tombstones(tombstone_ids)
        batches += 1
        processed += len(result.processed)
        stale += len(result.stale)
        if result.failed:
            raise RuntimeError("startup projection tombstone cleanup is retryable but incomplete")
        if not result.processed and not result.stale:
            raise RuntimeError("startup projection tombstone replay made no progress")


def _validate_startup_heads(source: SourceStore, relation: RelationStore) -> None:
    from memoryos.memory.canonical.current_head import artifact_root_for, iter_current_head_uris
    from memoryos.memory.canonical.visibility import read_committed_canonical

    artifact_root = artifact_root_for(source)
    if artifact_root is None:
        return
    for uri in iter_current_head_uris(
        artifact_root,
        kinds=("slot", "claim", "pending_proposal"),
    ):
        committed = read_committed_canonical(source, uri, relation)
        if committed.from_before_image:
            raise RuntimeError(f"current head and Source bundle disagree after recovery: {uri}")


def _validate_startup_canonical_domains(
    source: SourceStore,
    relation: RelationStore,
) -> dict[str, int]:
    """Rebuild every current Slot/Claim domain before the runtime can become READY."""

    from memoryos.memory.canonical.current_head import artifact_root_for, iter_current_head_uris

    artifact_root = artifact_root_for(source)
    if artifact_root is None:
        return {"slots": 0, "claims": 0}
    repository = CanonicalMemoryRepository(source, relation)
    slots = 0
    claims = 0
    slot_uris = set(iter_current_head_uris(artifact_root, kinds=("slot",)))
    claim_head_uris = set(iter_current_head_uris(artifact_root, kinds=("claim",)))
    claimed_head_uris: set[str] = set()
    for slot_uri in sorted(slot_uris):
        slot, slot_claims = repository.load_uri(slot_uri)
        member_uris = {f"{slot.uri}/claims/{claim_id}" for claim_id in slot.claim_ids}
        overlap = claimed_head_uris.intersection(member_uris)
        if overlap:
            raise RuntimeError("canonical Claim heads belong to multiple current Slots: " + ",".join(sorted(overlap)))
        claimed_head_uris.update(member_uris)
        slots += 1
        claims += len(slot_claims)
    if claim_head_uris != claimed_head_uris:
        detached = sorted(claim_head_uris - claimed_head_uris)
        missing = sorted(claimed_head_uris - claim_head_uris)
        raise RuntimeError(
            f"canonical Slot membership and current Claim heads disagree: detached={detached}; missing={missing}"
        )
    return {"slots": slots, "claims": claims}


def _validate_planning_salience_links(
    planning_store: PlanningEnvelopeStore,
    ledger: DurableSalienceLedger,
) -> dict[str, int]:
    validated = 0
    for envelope in planning_store.iter_payloads():
        task_id = str(envelope["task_id"])
        reservation = ledger.load(task_id)
        decision = dict(reservation["decision"])
        salience = dict(envelope.get("salience_decision", {}) or {})
        if (
            reservation["reservation_digest"] != envelope.get("salience_reservation_digest")
            or reservation["user_id"] != envelope.get("user_id")
            or decision.get("episode_fingerprint") != salience.get("episode_fingerprint")
            or int(decision.get("budget_cost", 0) or 0) != int(salience.get("budget_cost", 0) or 0)
            or bool(decision.get("duplicate", False)) != bool(salience.get("duplicate", False))
            or bool(decision.get("privacy_risk", False)) != bool(salience.get("privacy_risk", False))
        ):
            raise RuntimeError(f"planning envelope is detached from its salience reservation: {task_id}")
        validated += 1
    return {"validated": validated}


def _validate_commit_group_planning_links(
    service: SessionCommitService,
    planning_store: PlanningEnvelopeStore,
    ledger: DurableSalienceLedger,
) -> dict[str, int]:
    """Reverse-check semantic planning artifacts from every durable group."""

    validated = 0
    linked = 0
    allowed_phases = {
        "unstarted",
        "claimed",
        "salience_reserved",
        "planning_sealed",
        "committed",
    }
    for group in service.commit_group_store.all():
        if group.canonical_phase not in allowed_phases:
            raise RuntimeError(f"commit group has an invalid canonical phase: {group.group_id}")
        if group.tenant_id != planning_store.tenant_id:
            raise RuntimeError(f"commit group crosses runtime tenant: {group.group_id}")
        if group.group_id != f"commit_group_{group.task_id}":
            raise RuntimeError(f"commit group identity is detached from its task: {group.group_id}")
        requires_reservation = bool(
            group.salience_reservation_digest
            or group.canonical_phase in {"salience_reserved", "planning_sealed", "committed"}
            or group.canonical_status == "completed"
            or group.canonical_effects
        )
        reservation: dict[str, Any] | None = None
        if requires_reservation:
            reservation = ledger.load(group.task_id)
            if (
                reservation.get("reservation_digest") != group.salience_reservation_digest
                or reservation.get("user_id") != group.user_id
            ):
                raise RuntimeError(f"commit group is detached from its salience reservation: {group.group_id}")

        requires_envelope = bool(
            group.planning_digest
            or group.canonical_phase in {"planning_sealed", "committed"}
            or group.canonical_status == "completed"
            or group.canonical_effects
        )
        if requires_reservation or requires_envelope:
            archive = service.archive_store.read_archive(
                group.archive_uri,
                tenant_id=group.tenant_id,
                manifest_digest=group.manifest_digest or None,
            )
            if (
                archive.task_id != group.task_id
                or archive.user_id != group.user_id
                or archive.archive_digest != group.archive_digest
                or archive.manifest_digest != group.manifest_digest
            ):
                raise RuntimeError(f"commit group is detached from its immutable archive: {group.group_id}")
        if requires_envelope:
            envelope = planning_store.load_payload(group.task_id)
            if (
                envelope.get("operation_group_identity") != group.group_id
                or envelope.get("archive_uri") != group.archive_uri
                or envelope.get("archive_digest") != group.archive_digest
                or envelope.get("manifest_digest") != group.manifest_digest
                or envelope.get("user_id") != group.user_id
                or envelope.get("planning_digest") != group.planning_digest
                or envelope.get("salience_reservation_digest") != group.salience_reservation_digest
            ):
                raise RuntimeError(f"commit group is detached from its planning envelope: {group.group_id}")
            if reservation is None:
                raise RuntimeError(f"commit group planning has no salience reservation: {group.group_id}")
            linked += 1
        validated += 1
    return {"validated": validated, "linked": linked}


def _recover_startup_commit_groups(service: SessionCommitService) -> dict[str, Any]:
    store = service.commit_group_store
    abandoned = store.recover_abandoned_leases()
    expired = store.recover_expired_consumers()
    resumed: list[str] = []
    for status in store.pending():
        current = store.load(status.group_id)
        if current is None:
            raise RuntimeError(f"startup commit group disappeared: {status.group_id}")
        active = current.canonical_status == "running" or any(
            item.status == "running" for item in current.consumers.values()
        )
        if active:
            raise RuntimeError(f"startup commit group has a live lease: {status.group_id}")
        archive = service.archive_store.read_archive(
            current.archive_uri,
            tenant_id=current.tenant_id,
            manifest_digest=current.manifest_digest or None,
        )
        service.resume_startup_commit_group(archive, group_id=current.group_id)
        final = store.load(status.group_id)
        if final is None or not final.complete:
            canonical = final.canonical_status if final is not None else "missing"
            consumers = {name: item.status for name, item in final.consumers.items()} if final is not None else {}
            raise RuntimeError(
                f"startup commit group did not close: {status.group_id}; canonical={canonical}; consumers={consumers}"
            )
        resumed.append(status.group_id)
    return {
        "abandoned_leases": abandoned,
        "expired_leases": expired,
        "resumed": resumed,
    }


def _canonical_transaction_ids(group: Any) -> tuple[str, ...]:
    transaction_ids: list[str] = []
    for diff in group.canonical_effects.values():
        for operation in diff.get("operations", []) or []:
            if not isinstance(operation, dict):
                continue
            payload = operation.get("payload", {})
            if not isinstance(payload, dict) or payload.get("canonical_memory") is not True:
                continue
            transaction_id = str(payload.get("transaction_id") or "")
            if transaction_id and transaction_id not in transaction_ids:
                transaction_ids.append(transaction_id)
    return tuple(transaction_ids)


def _validate_commit_group_effect_links(
    service: SessionCommitService,
    committer: OperationCommitter,
) -> dict[str, int]:
    """Reverse-bind mutable commit-group progress to immutable memory receipts."""

    verified_groups = 0
    verified_effects = 0
    verified_operations = 0
    for group in service.commit_group_store.all():
        committed = committer.committed_memory_effect_diffs(group.user_id, group.group_id)
        expected_effects = {diff.diff_id: diff.to_dict() for diff in committed}
        if len(expected_effects) != len(committed):
            raise RuntimeError(f"commit group has duplicate immutable diff identities: {group.group_id}")
        actual_effects = dict(group.canonical_effects)
        if set(actual_effects) != set(expected_effects) or any(
            canonical_digest(actual_effects[diff_id]) != canonical_digest(expected)
            for diff_id, expected in expected_effects.items()
        ):
            raise RuntimeError(f"commit group effects are detached from immutable receipts: {group.group_id}")

        expected_operations: dict[str, dict[str, Any]] = {}
        for diff in expected_effects.values():
            for operation in diff.get("operations", []) or []:
                if not isinstance(operation, dict):
                    raise RuntimeError(f"commit group immutable effect is malformed: {group.group_id}")
                operation_id = str(operation.get("operation_id") or "")
                if not operation_id:
                    raise RuntimeError(f"commit group immutable effect has no operation id: {group.group_id}")
                existing = expected_operations.get(operation_id)
                if existing is not None and canonical_digest(existing) != canonical_digest(operation):
                    raise RuntimeError(f"commit group reuses an operation id: {group.group_id}")
                expected_operations[operation_id] = operation

        if group.canonical_status == "completed":
            if group.canonical_phase != "committed":
                raise RuntimeError(f"completed commit group has no committed phase: {group.group_id}")
            result_operations = group.canonical_result.get("operations", []) or []
            if not isinstance(result_operations, list):
                raise RuntimeError(f"commit group canonical result is malformed: {group.group_id}")
            actual_operations: dict[str, dict[str, Any]] = {}
            for operation in result_operations:
                if not isinstance(operation, dict):
                    raise RuntimeError(f"commit group canonical result operation is malformed: {group.group_id}")
                operation_id = str(operation.get("operation_id") or "")
                if not operation_id or operation_id in actual_operations:
                    raise RuntimeError(f"commit group canonical result has duplicate operations: {group.group_id}")
                actual_operations[operation_id] = operation
            if set(actual_operations) != set(expected_operations) or any(
                canonical_digest(actual_operations[operation_id]) != canonical_digest(expected)
                for operation_id, expected in expected_operations.items()
            ):
                raise RuntimeError(
                    f"commit group canonical result is detached from immutable receipts: {group.group_id}"
                )
            if int(group.canonical_result.get("operation_count", len(actual_operations))) != len(actual_operations):
                raise RuntimeError(f"commit group canonical operation count is corrupt: {group.group_id}")
            revisions = []
            for operation in expected_operations.values():
                context_object = dict(operation.get("payload", {}) or {}).get("context_object")
                if not isinstance(context_object, dict):
                    continue
                revision = dict(context_object.get("metadata", {}) or {}).get("revision")
                if revision is not None:
                    revisions.append(int(revision))
            expected_revision = max(revisions) if revisions else None
            if group.canonical_revision != expected_revision:
                raise RuntimeError(f"commit group canonical revision is corrupt: {group.group_id}")

        verified_groups += 1
        verified_effects += len(expected_effects)
        verified_operations += len(expected_operations)
    return {
        "verified_groups": verified_groups,
        "verified_effects": verified_effects,
        "verified_operations": verified_operations,
    }


def _validate_completed_projection_consumers(
    service: SessionCommitService,
    worker: MemoryProjectionWorker,
) -> dict[str, int]:
    """Reverse-prove terminal projection consumers from immutable transactions."""

    verified_groups = 0
    verified_transactions = 0
    refreshed = 0
    migrated_legacy_proofs = 0
    for group in service.commit_group_store.all():
        projection = group.consumers["projection"]
        if projection.status != "completed":
            continue
        transaction_ids = _canonical_transaction_ids(group)
        if not transaction_ids:
            # A salience skip or a group containing no canonical-memory effect
            # has no projection transaction to prove.
            verified_groups += 1
            continue
        raw_legacy = projection.result.get("completion_proofs", [])
        legacy_proofs = (
            {
                str(item.get("transaction_id") or ""): item
                for item in raw_legacy
                if isinstance(item, dict) and item.get("schema_version") == "projection_completion_proof_v1"
            }
            if isinstance(raw_legacy, list)
            else {}
        )
        for transaction_id in transaction_ids:
            if worker.proof_store.load_publication(transaction_id) is not None:
                continue
            legacy = legacy_proofs.get(transaction_id)
            if legacy is not None and worker.migrate_legacy_completion_proof(
                group.group_id,
                transaction_id,
                legacy,
            ):
                migrated_legacy_proofs += 1
        completion = worker.verify_commit_group_completion(group.group_id, transaction_ids)
        failures = [str(item) for item in completion["failures"]]
        if failures:
            raise RuntimeError(f"completed projection consumer lacks durable proof: {group.group_id}: {failures}")
        proof_result = {
            "status": "completed",
            "transaction_ids": list(transaction_ids),
            "completion_proofs": list(completion["proofs"]),
        }
        if projection.result != proof_result:
            service.commit_group_store.refresh_completed_consumer_result(
                group.group_id,
                "projection",
                result=proof_result,
            )
            refreshed += 1
        verified_groups += 1
        verified_transactions += len(transaction_ids)
    return {
        "verified_groups": verified_groups,
        "verified_transactions": verified_transactions,
        "refreshed": refreshed,
        "migrated_legacy_proofs": migrated_legacy_proofs,
    }
