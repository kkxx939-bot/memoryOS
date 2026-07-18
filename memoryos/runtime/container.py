"""Greenfield runtime composition for user-editable Markdown memory."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from memoryos.action_policy.integration.commit_registration import build_action_policy_commit_handlers
from memoryos.adapters.agent_hooks.session_service import AgentSessionService
from memoryos.adapters.persistence.filesystem import FileSystemSourceStore
from memoryos.adapters.persistence.filesystem.memory_document_store import FileSystemMemoryDocumentStore
from memoryos.adapters.persistence.filesystem.session_archive import SessionArchiveStore
from memoryos.adapters.persistence.sqlite import (
    SQLiteIndexStore,
    SQLiteLockStore,
    SQLiteQueueStore,
    SQLiteRelationStore,
)
from memoryos.application.context.maintenance import (
    CallbackDocumentServingMaintenance,
    CatalogDocumentProjectionVerifier,
    DerivedServingMaintenanceService,
)
from memoryos.application.context.reranking import Reranker
from memoryos.application.context.trace_erase import RecallTraceEraseBackend
from memoryos.application.memory.command_service import MemoryCommandService
from memoryos.application.memory.pending_review_service import MemoryEditReviewService
from memoryos.application.session.commit_group import CommitGroupStore
from memoryos.application.session.commit_service import SessionCommitService
from memoryos.application.session.context_projector import SessionContextProjector
from memoryos.application.session.planners import ActionPolicyCommitPlanner, BehaviorCommitPlanner, MemoryCommitPlanner
from memoryos.application.session.projection_journal import SessionProjectionJournal
from memoryos.contextdb.catalog import CatalogRecordKind
from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.retention import CatalogRetentionManager, RetentionPolicy
from memoryos.contextdb.retrieval.embedding import EmbeddingProvider
from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
from memoryos.contextdb.session.evidence_encoder import register_session_evidence_encoder
from memoryos.contextdb.store.index_store import IndexStore
from memoryos.contextdb.store.lock_store import LockStore
from memoryos.contextdb.store.queue_store import QueueStore
from memoryos.contextdb.store.relation_store import RelationStore
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.contextdb.store.vector import VectorStore, require_production_vector_capabilities
from memoryos.contextdb.tombstone import ProjectionTombstoneService
from memoryos.contextdb.transaction.path_lock import PathLock
from memoryos.core.readiness import RuntimeReadiness, RuntimeReadinessState
from memoryos.execution.action_executor import ActionExecutor
from memoryos.execution.tool_registry import ToolRegistry
from memoryos.memory.documents import (
    DocumentAdoptionReceipt,
    DocumentCommitResult,
    DocumentDeletionStatus,
    DocumentEditKind,
    ExternalChangeKind,
    ManagedDocument,
    MemoryDocumentBootstrapper,
    MemoryDocumentCommitter,
    MemoryDocumentConsolidationStore,
    MemoryDocumentConsolidator,
    MemoryDocumentContextOverlay,
    MemoryDocumentControlStore,
    MemoryDocumentEraser,
    MemoryDocumentPlanner,
    MemoryDocumentProjector,
    MemoryDocumentRevisionStore,
    MemoryDocumentScanner,
    MemoryEditProposal,
    MemoryEditReviewStore,
    PresentPath,
    RelatedDocumentCandidate,
    RuntimeLayout,
    matches_adopted_source,
    validate_document_id,
)
from memoryos.memory.documents.path_policy import MemoryDocumentPathPolicy
from memoryos.memory.documents.scanner import ExternalDocumentChange
from memoryos.memory.evidence import (
    DurableSalienceLedger,
    SealedProposalEraseBackend,
    SealedProposalStore,
    SessionEvidenceArchiveEncoder,
)
from memoryos.operations.commit.domain_registry import register_action_policy_commit_handlers
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.commit.recovery import RecoveryService
from memoryos.prediction.model.prediction_ledger import PredictionLedger
from memoryos.prediction.pipeline.prediction_engine import PredictionEngine
from memoryos.runtime.config import RuntimeConfig
from memoryos.workers.memory_document_edit_worker import MemoryDocumentEditWorker
from memoryos.workers.memory_document_projection_worker import (
    MemoryDocumentCatalogEraseBackend,
    MemoryDocumentProjectionWorker,
)
from memoryos.workers.memory_document_scan_worker import MemoryDocumentScanWorker
from memoryos.workers.recovery_worker import RecoveryWorker

_MAX_STARTUP_MEMORY_OWNERS = 1_000


@dataclass
class RuntimeContainer:
    """One explicit composition shared by SDK, transports, and workers."""

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
    memory_document_store: FileSystemMemoryDocumentStore
    memory_document_control_store: MemoryDocumentControlStore
    memory_document_revision_store: MemoryDocumentRevisionStore
    memory_document_bootstrapper: MemoryDocumentBootstrapper
    memory_document_planner: MemoryDocumentPlanner
    memory_document_committer: MemoryDocumentCommitter
    memory_document_consolidation_store: MemoryDocumentConsolidationStore
    memory_document_consolidator: MemoryDocumentConsolidator
    memory_document_projector: MemoryDocumentProjector
    memory_document_scanner: MemoryDocumentScanner
    memory_document_edit_worker: MemoryDocumentEditWorker
    memory_document_scan_worker: MemoryDocumentScanWorker
    memory_projection_worker: MemoryDocumentProjectionWorker
    memory_document_eraser: MemoryDocumentEraser
    memory_command_service: MemoryCommandService
    memory_review_service: MemoryEditReviewService
    recovery_service: RecoveryService
    recovery_worker: RecoveryWorker
    readiness: RuntimeReadiness
    agent_session_service: AgentSessionService
    tombstone_service: ProjectionTombstoneService
    retention_manager: CatalogRetentionManager


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
    """Build the Markdown-only runtime and recover it before publishing READY.

    Layout validation deliberately precedes construction of every SQLite
    adapter.  An unmarked or legacy root therefore cannot be mutated by an
    attempted migration or by database initialization.
    """

    root = config.root_path
    layout = RuntimeLayout.open(root, tenant_id=config.tenant_id)
    layout_details = layout.initialize_or_validate()
    tenant_root = layout.tenant_root

    readiness = RuntimeReadiness()
    readiness.transition(
        RuntimeReadinessState.RECOVERING,
        details={"runtime_layout": layout_details},
    )
    document_store = FileSystemMemoryDocumentStore(
        root,
        max_file_bytes=config.memory_document_max_bytes,
        max_front_matter_bytes=config.memory_front_matter_max_bytes,
        max_front_matter_depth=config.memory_front_matter_max_depth,
        max_scan_files=config.memory_scan_max_files,
    )
    # Probe the filesystem before constructing SQLite serving state.  A root
    # that cannot honor the exact-byte durability primitives must fail closed
    # without leaving a partially initialized database set behind.
    document_store.probe_write_capabilities(config.tenant_id)
    source = source_store or FileSystemSourceStore(root, tenant_id=config.tenant_id)
    source_tenant = str(getattr(source, "tenant_id", config.tenant_id))
    if source_tenant != config.tenant_id:
        raise ValueError("SourceStore tenant does not match RuntimeConfig tenant_id")
    if hasattr(source, "__dict__"):
        vars(source)["readiness"] = readiness

    index_root = tenant_root / "indexes"
    index = index_store or SQLiteIndexStore(index_root / "context.sqlite3")
    relation = relation_store or SQLiteRelationStore(index_root / "relations.sqlite3")
    queue = queue_store or SQLiteQueueStore(tenant_root / "queues" / "jobs.sqlite3")
    lock = lock_store or SQLiteLockStore(tenant_root / "system" / "locks.sqlite3")

    configured_vector = vector_store or config.vector_store
    configured_embedding = embedding_provider or config.embedding
    if config.mode == "server" and configured_vector is not None:
        require_production_vector_capabilities(configured_vector)
    search = hybrid_search or HybridSearch(
        index,
        vector_store=configured_vector,
        embedding_provider=configured_embedding,
        source_store=source,
    )

    register_session_evidence_encoder(SessionEvidenceArchiveEncoder())
    register_action_policy_commit_handlers(build_action_policy_commit_handlers())
    committer = OperationCommitter(
        source,
        index,
        str(root),
        lock_store=lock,
        relation_store=relation,
        queue_store=queue,
        tenant_id=config.tenant_id,
    )

    control_store = MemoryDocumentControlStore(root)
    revision_store = MemoryDocumentRevisionStore(root, max_blob_bytes=config.memory_document_max_bytes)
    review_store = MemoryEditReviewStore(root, max_blob_bytes=config.memory_document_max_bytes)
    bootstrapper = MemoryDocumentBootstrapper(
        root,
        document_store,
        control_store=control_store,
        max_front_matter_bytes=config.memory_front_matter_max_bytes,
        max_front_matter_depth=config.memory_front_matter_max_depth,
    )
    document_planner = MemoryDocumentPlanner(
        document_store,
        max_front_matter_bytes=config.memory_front_matter_max_bytes,
        max_front_matter_depth=config.memory_front_matter_max_depth,
        max_edit_bytes=config.memory_document_max_bytes,
        related_document_finder=lambda tenant, owner, proposal, limit: _find_related_memory_documents(
            index,
            tenant_id=tenant,
            owner_user_id=owner,
            proposal=proposal,
            limit=limit,
        ),
        max_related_documents=8,
    )
    document_committer = MemoryDocumentCommitter(
        document_store,
        control_store,
        revision_store,
        queue,
        path_lock=PathLock(lock),
    )
    document_projector = MemoryDocumentProjector(
        max_front_matter_bytes=config.memory_front_matter_max_bytes,
        max_front_matter_depth=config.memory_front_matter_max_depth,
    )
    projection_worker = MemoryDocumentProjectionWorker(
        document_store,
        control_store,
        index,  # type: ignore[arg-type]
        queue,
        projector=document_projector,
        vector_store=configured_vector,
        embedding_provider=configured_embedding,
        relation_store=relation,
    )
    consolidation_store = MemoryDocumentConsolidationStore(root)
    consolidator = MemoryDocumentConsolidator(
        document_committer,
        index,  # type: ignore[arg-type]
        saga_store=consolidation_store,
    )
    sealed_proposal_store = SealedProposalStore(root, tenant_id=config.tenant_id)
    document_eraser = MemoryDocumentEraser(
        document_store,
        control_store,
        revision_store,
        review_store=review_store,
        cleanup_backends=(
            MemoryDocumentCatalogEraseBackend(projection_worker),
            SealedProposalEraseBackend(sealed_proposal_store),
            RecallTraceEraseBackend(root),
        ),
    )
    command_service = MemoryCommandService(
        document_planner,
        document_committer,
        document_eraser,
        bootstrapper=bootstrapper,
        independent_evidence_locator=lambda tenant, owner, document, _digest: (
            _independent_session_archives(
                control_store.lineage_references(tenant, owner, document)
            )
        ),
        readiness=readiness,
        consolidator=consolidator,
        review_store=review_store,
    )
    review_service = MemoryEditReviewService(
        review_store,
        document_committer,
        readiness=readiness,
        consolidator=consolidator,
    )
    def publish_external_change(change: ExternalDocumentChange) -> None:
        _publish_external_change(
            change,
            committer=document_committer,
            control_store=control_store,
            document_store=document_store,
            bootstrapper=bootstrapper,
        )

    scanner = MemoryDocumentScanner(
        document_store,
        control_store=control_store,
        stability_seconds=config.memory_scan_stability_seconds,
        mass_delete_threshold=config.memory_mass_delete_threshold,
        change_publisher=publish_external_change,
    )
    document_edit_worker = MemoryDocumentEditWorker(
        document_committer,
        queue,
        tenant_id=config.tenant_id,
        readiness=readiness,
    )
    document_scan_worker = MemoryDocumentScanWorker(
        scanner,
        queue,
        tenant_id=config.tenant_id,
        owner_user_ids=lambda tenant, limit: _bounded_owner_ids(
            layout,
            tenant,
            limit,
        ),
        owner_enumeration_limit=_MAX_STARTUP_MEMORY_OWNERS,
        readiness=readiness,
    )
    document_overlay = MemoryDocumentContextOverlay(
        document_store,
        max_front_matter_bytes=config.memory_front_matter_max_bytes,
        max_front_matter_depth=config.memory_front_matter_max_depth,
    )

    archive_store = SessionArchiveStore(root, tenant_id=config.tenant_id)
    session_projector = SessionContextProjector(
        index,  # type: ignore[arg-type]
        vector_store=configured_vector,
        embedding_provider=configured_embedding,
        vectorize_important_events=bool(
            dict(config.retrieval or {}).get("vectorize_important_session_events", False)
        ),
    )
    memory_planner = MemoryCommitPlanner(
        document_planner,
        extractor=config.memory_extractor,
        archive_store=archive_store,
        salience_ledger=DurableSalienceLedger(root, tenant_id=config.tenant_id),
        bootstrapper=bootstrapper,
        proposal_store=sealed_proposal_store,
        review_store=review_store,
        tenant_id=config.tenant_id,
    )
    session_commit_service = SessionCommitService(
        archive_store,
        queue,
        committer=committer,
        memory_planner=memory_planner,
        behavior_planner=BehaviorCommitPlanner(index_store=index, source_store=source),
        action_policy_planner=ActionPolicyCommitPlanner(index_store=index, source_store=source),
        session_projector=session_projector,
        commit_group_store=CommitGroupStore(tenant_root),
        memory_committer=document_committer,
        document_planner=document_planner,
        projection_journal=SessionProjectionJournal(index),
    )

    tombstone_service = ProjectionTombstoneService(
        index,
        source_store=source,
        vector_store=configured_vector,
        relation_store=relation,
    )
    committer.tombstone_service = tombstone_service
    retention_manager = CatalogRetentionManager(
        index,  # type: ignore[arg-type]
        vector_store=configured_vector,
        tombstone_service=tombstone_service,
        policy=RetentionPolicy.from_config(config.retention),
    )
    context_db = ContextDB(
        source,
        index,
        relation,
        queue_store=queue,
        session_commit_service=session_commit_service,
        committer=committer,
        document_overlay=document_overlay,
        tombstone_service=tombstone_service,
        retention_manager=retention_manager,
        readiness=readiness,
        tenant_id=config.tenant_id,
    )
    document_serving = CallbackDocumentServingMaintenance(
        full_scan=document_store.full_scan,
        rebuild_owner=projection_worker.rebuild_owner,
        verify_owner=CatalogDocumentProjectionVerifier(index),
        owner_user_ids=lambda tenant, limit: _bounded_owner_ids(layout, tenant, limit),
        max_documents_per_owner=config.memory_scan_max_files,
    )
    context_db._configure_extensions(
        administration_service=DerivedServingMaintenanceService(
            source,
            index,
            relation,
            tenant_id=config.tenant_id,
            document_serving=document_serving,
            retention_manager=retention_manager,
            readiness=readiness,
            domain_overlay=context_db.domain_overlay,
            index_policy=context_db.index_policy,
            serving_lock=context_db.serving_lock,
        )
    )

    recovery_service = RecoveryService(committer.redo, committer)
    recovery_worker = RecoveryWorker(recovery_service)
    registry = tool_registry or ToolRegistry()
    engine = PredictionEngine(
        index,
        PredictionLedger(root),
        source_store=source,
        relation_store=relation,
        vector_store=configured_vector,
        embedding_provider=configured_embedding,
        hybrid_search=search,
    )
    agent_session_service = AgentSessionService(str(root), tenant_id=config.tenant_id)

    container = RuntimeContainer(
        source_store=source,
        index_store=index,
        relation_store=relation,
        queue_store=queue,
        lock_store=lock,
        vector_store=configured_vector,
        embedding_provider=configured_embedding,
        hybrid_search=search,
        reranker=config.reranker,
        committer=committer,
        session_archive_store=archive_store,
        session_commit_service=session_commit_service,
        context_db=context_db,
        engine=engine,
        executor=ActionExecutor(registry),
        memory_document_store=document_store,
        memory_document_control_store=control_store,
        memory_document_revision_store=revision_store,
        memory_document_bootstrapper=bootstrapper,
        memory_document_planner=document_planner,
        memory_document_committer=document_committer,
        memory_document_consolidation_store=consolidation_store,
        memory_document_consolidator=consolidator,
        memory_document_projector=document_projector,
        memory_document_scanner=scanner,
        memory_document_edit_worker=document_edit_worker,
        memory_document_scan_worker=document_scan_worker,
        memory_projection_worker=projection_worker,
        memory_document_eraser=document_eraser,
        memory_command_service=command_service,
        memory_review_service=review_service,
        recovery_service=recovery_service,
        recovery_worker=recovery_worker,
        readiness=readiness,
        agent_session_service=agent_session_service,
        tombstone_service=tombstone_service,
        retention_manager=retention_manager,
    )
    _recover_runtime(container, layout=layout)
    return container


def _find_related_memory_documents(
    index_store: IndexStore,
    *,
    tenant_id: str,
    owner_user_id: str,
    proposal: MemoryEditProposal,
    limit: int,
) -> tuple[RelatedDocumentCandidate, ...]:
    """Return bounded Catalog hints; the planner revalidates exact live bytes."""

    tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
    owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
    maximum = max(1, min(int(limit), 8))
    search_catalog = getattr(index_store, "search_catalog", None)
    if not callable(search_catalog):
        return ()
    semantic_query = "\n".join(
        value.strip()
        for value in (proposal.title, proposal.subject, proposal.body)
        if value and value.strip()
    )[:4_096]
    raw_hits = search_catalog(
        semantic_query,
        tenant_id=tenant,
        filters={
            "tenant_id": tenant,
            "owner_user_id": owner,
            "record_kinds": (CatalogRecordKind.MEMORY_DOCUMENT.value,),
            "lifecycle_state": "active",
        },
        limit=maximum,
    )
    if not isinstance(raw_hits, Sequence) or isinstance(raw_hits, (str, bytes)):
        raise TypeError("related memory Catalog search returned an invalid result")
    hits = tuple(raw_hits)
    if len(hits) > maximum:
        raise RuntimeError("related memory Catalog search exceeded its bound")
    candidates: list[RelatedDocumentCandidate] = []
    seen: set[tuple[str, str]] = set()
    for hit in hits:
        metadata = dict(getattr(hit, "metadata", {}) or {})
        if (
            str(metadata.get("tenant_id") or "") != tenant
            or str(metadata.get("owner_user_id") or "") != owner
            or str(metadata.get("record_kind") or "")
            != CatalogRecordKind.MEMORY_DOCUMENT.value
            or str(metadata.get("lifecycle_state") or "") != "active"
        ):
            raise PermissionError("related memory Catalog search crossed its bounded scope")
        try:
            document_id = validate_document_id(str(metadata.get("document_id") or ""))
            relative_path = MemoryDocumentPathPolicy.normalize_relative_path(
                str(metadata.get("relative_path") or "")
            )
        except ValueError:
            continue
        source_digest = str(metadata.get("source_digest") or "")
        if len(source_digest) != 64 or any(
            character not in "0123456789abcdef" for character in source_digest
        ):
            continue
        identity = (document_id, relative_path)
        if identity in seen:
            continue
        seen.add(identity)
        candidates.append(
            RelatedDocumentCandidate(
                tenant_id=tenant,
                owner_user_id=owner,
                document_id=document_id,
                relative_path=relative_path,
                source_digest=source_digest,
                relevance=float(getattr(hit, "score", 0.0) or 0.0),
            )
        )
    return tuple(candidates)


def _publish_external_change(
    change: ExternalDocumentChange,
    *,
    committer: MemoryDocumentCommitter,
    control_store: MemoryDocumentControlStore,
    document_store: FileSystemMemoryDocumentStore,
    bootstrapper: MemoryDocumentBootstrapper,
) -> DocumentCommitResult | None:
    """Publish scanner facts and close an adopt-first bootstrap before READY."""

    result = committer.record_external_change(change)
    if change.change_kind.value != "create":
        return result
    receipt = control_store.load_adoption_receipt_for_document(
        change.tenant_id,
        change.owner_user_id,
        change.document_id,
    )
    if receipt is None:
        return result
    raw = document_store.read_raw(
        change.tenant_id,
        change.owner_user_id,
        document_id=change.document_id,
    )
    exact_adoption = (
        change.new_relative_path == receipt.relative_path
        and hashlib.sha256(raw).hexdigest() == change.after_raw_digest
        and matches_adopted_source(raw, receipt.document_id, receipt.expected_raw_sha256)
    )
    if exact_adoption:
        bootstrapper.ensure_adopted_user(
            change.tenant_id,
            change.owner_user_id,
            receipt.relative_path,
            document_id=receipt.document_id,
            adopted_raw_sha256=change.after_raw_digest,
        )
    else:
        # A completed marker makes later edits/renames ordinary. Without one,
        # fail startup closed instead of leaving remember unusable after READY.
        bootstrapper.ensure_user(change.tenant_id, change.owner_user_id)
    return result


def _recover_adoption_receipts(
    container: RuntimeContainer,
    *,
    tenant_id: str,
    owners: tuple[str, ...],
) -> dict[str, Any]:
    """Resume receipt-authorized source CAS before the ordinary scanner runs.

    A receipt can become durable before the unmanaged source is replaced, or
    the replacement can become durable before its CREATE intent/event.  Those
    are the only two body states accepted here.  Everything else is preserved
    as an observable startup conflict.
    """

    totals = {
        "receipts": 0,
        "already_committed": 0,
        "bootstrap_resumed": 0,
        "erasure_blocked": 0,
        "resumed_unmanaged": 0,
        "resumed_managed": 0,
        "published": 0,
    }
    per_owner: dict[str, dict[str, int]] = {}
    for owner in owners:
        owner_counts = {key: 0 for key in totals}
        receipts = container.memory_document_control_store.adoption_receipts(
            tenant_id,
            owner,
        )
        owner_counts["receipts"] = len(receipts)
        active: list[DocumentAdoptionReceipt] = []
        active_paths: set[str] = set()
        active_document_ids: set[str] = set()
        for receipt in receipts:
            if receipt.tenant_id != tenant_id or receipt.owner_user_id != owner:
                raise RuntimeError("adoption receipt enumeration crossed its exact scope")
            indexed_receipt = (
                container.memory_document_control_store.load_adoption_receipt_for_document(
                    tenant_id,
                    owner,
                    receipt.document_id,
                )
            )
            if indexed_receipt is not None and indexed_receipt != receipt:
                raise RuntimeError("adoption receipt identity index changed its exact authority")
            erase_record = container.memory_document_committer.erasure_store.load(
                tenant_id,
                owner,
                receipt.document_id,
            )
            barrier = container.memory_document_control_store.load_publication_barrier(
                tenant_id,
                owner,
                receipt.document_id,
            )
            if erase_record is not None:
                # Every erasure epoch, including a retryable pending one, is
                # durable anti-resurrection authority for this identity.
                if (
                    indexed_receipt != receipt
                    or barrier is None
                    or barrier.status is not DocumentDeletionStatus.HARD_ERASED
                ):
                    raise RuntimeError("adoption erasure is detached from its durable identity barrier")
                owner_counts["erasure_blocked"] += 1
                continue
            if barrier is not None and barrier.status is DocumentDeletionStatus.HARD_ERASED:
                raise RuntimeError("hard-erased adoption identity is missing its durable erasure epoch")
            control = container.memory_document_control_store.load_control(
                tenant_id,
                owner,
                receipt.document_id,
            )
            if control is not None:
                if indexed_receipt != receipt:
                    raise RuntimeError("committed adoption is missing its exact receipt identity index")
                owner_counts["already_committed"] += 1
                bootstrap_status = container.memory_document_bootstrapper.status(
                    tenant_id,
                    owner,
                )
                if bootstrap_status == "COMPLETED":
                    continue
                binding = container.memory_document_control_store.load_event_binding(
                    tenant_id,
                    owner,
                    receipt.document_id,
                    control.last_event_id,
                )
                if binding is None:
                    raise RuntimeError("committed adoption is missing its durable CREATE event")
                intent, event = binding
                if (
                    control.status != "present"
                    or control.relative_path != receipt.relative_path
                    or event.edit_kind is not DocumentEditKind.CREATE
                    or event.old_relative_path
                    or event.new_relative_path != receipt.relative_path
                    or event.after_raw_digest != control.raw_sha256
                    or event.logical_revision != control.logical_revision
                    or event.projection_generation != control.projection_generation
                    or event.actor_binding != receipt.actor_binding
                    or event.evidence_reference != receipt.evidence_reference
                    or event.evidence_digest != receipt.evidence_digest
                    or event.edit_summary != receipt.edit_summary
                    or intent.idempotency_digest
                    != hashlib.sha256(receipt.idempotency_key.encode()).hexdigest()
                ):
                    raise RuntimeError("committed adoption is detached from its exact receipt lineage")
                live = container.memory_document_store.read_state(
                    tenant_id,
                    owner,
                    receipt.relative_path,
                )
                if not isinstance(live, PresentPath) or live.raw_sha256 != control.raw_sha256:
                    raise RuntimeError("unbootstrapped adoption no longer matches its durable control")
                raw = container.memory_document_store.read_raw(
                    tenant_id,
                    owner,
                    relative_path=receipt.relative_path,
                )
                if (
                    hashlib.sha256(raw).hexdigest() != control.raw_sha256
                    or not matches_adopted_source(
                        raw,
                        receipt.document_id,
                        receipt.expected_raw_sha256,
                    )
                ):
                    raise RuntimeError("unbootstrapped adoption is not the exact receipt rewrite")
                scan = container.memory_document_store.full_scan(tenant_id, owner)
                exact = [
                    registration
                    for registration in scan.registrations
                    if isinstance(registration, ManagedDocument)
                    and registration.document_id == receipt.document_id
                    and registration.relative_path == receipt.relative_path
                    and registration.raw_sha256 == control.raw_sha256
                ]
                if not scan.complete or scan.errors or len(exact) != 1:
                    raise RuntimeError("unbootstrapped adoption is unsafe or duplicated")
                container.memory_document_bootstrapper.ensure_adopted_user(
                    tenant_id,
                    owner,
                    receipt.relative_path,
                    document_id=receipt.document_id,
                    adopted_raw_sha256=control.raw_sha256,
                )
                if container.memory_document_bootstrapper.status(tenant_id, owner) != "COMPLETED":
                    raise RuntimeError("committed adoption bootstrap did not reach COMPLETED")
                owner_counts["bootstrap_resumed"] += 1
                continue
            if receipt.relative_path in active_paths:
                raise RuntimeError("active adoption receipts duplicate one relative path")
            if receipt.document_id in active_document_ids:
                raise RuntimeError("active adoption receipts duplicate one document identity")
            active_paths.add(receipt.relative_path)
            active_document_ids.add(receipt.document_id)

            # A stop can occur between receipt publication and its content-free
            # document-ID index.  Replay the exact receipt to repair only that
            # immutable index before the committer consumes it.
            container.memory_document_committer.verify_adoption_root(
                tenant_id,
                owner,
                receipt.document_id,
            )
            durable = container.memory_document_control_store.prepare_adoption_receipt(
                tenant_id,
                owner,
                receipt.relative_path,
                receipt.expected_raw_sha256,
                actor_binding=receipt.actor_binding,
            )
            if durable != receipt:
                raise RuntimeError("adoption receipt replay changed its durable authority")
            active.append(receipt)

        for receipt in active:
            state = container.memory_document_store.read_state(
                tenant_id,
                owner,
                receipt.relative_path,
            )
            if not isinstance(state, PresentPath):
                raise RuntimeError("adoption receipt target is absent or unsafe during startup")
            raw = container.memory_document_store.read_raw(
                tenant_id,
                owner,
                relative_path=receipt.relative_path,
            )
            raw_digest = hashlib.sha256(raw).hexdigest()
            if raw_digest != state.raw_sha256:
                raise RuntimeError("adoption receipt target changed during startup classification")
            if raw_digest == receipt.expected_raw_sha256:
                adopted = container.memory_document_store.adopt(
                    tenant_id,
                    owner,
                    receipt.relative_path,
                    expected_raw_sha256=receipt.expected_raw_sha256,
                    assigned_document_id=receipt.document_id,
                    operation_id=receipt.receipt_id,
                )
                if (
                    adopted.document_id != receipt.document_id
                    or adopted.relative_path != receipt.relative_path
                    or not matches_adopted_source(
                        adopted.raw_bytes,
                        receipt.document_id,
                        receipt.expected_raw_sha256,
                    )
                ):
                    raise RuntimeError("resumed adoption produced bytes detached from its receipt")
                owner_counts["resumed_unmanaged"] += 1
            elif matches_adopted_source(
                raw,
                receipt.document_id,
                receipt.expected_raw_sha256,
            ):
                owner_counts["resumed_managed"] += 1
            else:
                raise RuntimeError("adoption receipt target is a third source state")

        if active:
            scan = container.memory_document_store.full_scan(tenant_id, owner)
            if not scan.complete or scan.errors:
                raise RuntimeError("resumed adoption requires one complete registration scan")
            for receipt in active:
                registrations = [
                    registration
                    for registration in scan.registrations
                    if isinstance(registration, ManagedDocument)
                    and registration.document_id == receipt.document_id
                    and registration.relative_path == receipt.relative_path
                ]
                if len(registrations) != 1:
                    raise RuntimeError("resumed adoption is unsafe, duplicated, or unregistered")
                managed = registrations[0]
                raw = container.memory_document_store.read_raw(
                    tenant_id,
                    owner,
                    document_id=receipt.document_id,
                )
                if (
                    hashlib.sha256(raw).hexdigest() != managed.raw_sha256
                    or not matches_adopted_source(
                        raw,
                        receipt.document_id,
                        receipt.expected_raw_sha256,
                    )
                ):
                    raise RuntimeError("resumed adoption no longer matches its exact source receipt")
                _publish_external_change(
                    ExternalDocumentChange(
                        change_kind=ExternalChangeKind.CREATE,
                        tenant_id=tenant_id,
                        owner_user_id=owner,
                        document_id=receipt.document_id,
                        old_relative_path="",
                        new_relative_path=receipt.relative_path,
                        before_raw_digest="",
                        after_raw_digest=managed.raw_sha256,
                        scan_generation_id=scan.generation_id,
                    ),
                    committer=container.memory_document_committer,
                    control_store=container.memory_document_control_store,
                    document_store=container.memory_document_store,
                    bootstrapper=container.memory_document_bootstrapper,
                )
                control = container.memory_document_control_store.load_control(
                    tenant_id,
                    owner,
                    receipt.document_id,
                )
                if (
                    control is None
                    or control.status != "present"
                    or control.relative_path != receipt.relative_path
                    or control.raw_sha256 != managed.raw_sha256
                    or container.memory_document_control_store.load_event_binding(
                        tenant_id,
                        owner,
                        receipt.document_id,
                        control.last_event_id,
                    )
                    is None
                ):
                    raise RuntimeError("resumed adoption did not publish its exact control and event")
                owner_counts["published"] += 1

        per_owner[owner] = owner_counts
        for key in totals:
            totals[key] += owner_counts[key]
    return {**totals, "owners": per_owner}


def _recover_runtime(container: RuntimeContainer, *, layout: RuntimeLayout) -> None:
    details: dict[str, Any] = {"runtime_layout": "markdown_memory_v1"}
    try:
        details["queue_expired_leases"] = container.queue_store.recover_expired_leases()
        details["ordinary_operations"] = container.recovery_worker.process_all()
        owners = _bounded_owner_ids(
            layout,
            layout.tenant_id,
            _MAX_STARTUP_MEMORY_OWNERS,
        )
        details["owners"] = list(owners)
        for owner in owners:
            container.memory_document_store.probe_write_capabilities(
                layout.tenant_id,
                owner,
            )
        details["memory_source_filesystems_probed"] = len(owners)
        document_recovery: dict[str, Any] = {}
        for owner in owners:
            report = container.memory_document_committer.recover_all(layout.tenant_id, owner)
            if report.conflicted_intent_ids:
                raise RuntimeError(
                    "document recovery preserved third-state external edits: "
                    + ",".join(report.conflicted_intent_ids)
                )
            document_recovery[owner] = {"completed": len(report.completed), "conflicted": 0}
        details["document_intents"] = document_recovery
        erasure_recovery: dict[str, Any] = {}
        for owner in owners:
            erasure_report = container.memory_document_eraser.recover_owner(layout.tenant_id, owner)
            erasure_recovery[owner] = {
                "completed": list(erasure_report.completed_document_ids),
                "pending": list(erasure_report.pending_document_ids),
            }
        details["memory_document_erasures"] = erasure_recovery
        details["memory_document_adoptions"] = _recover_adoption_receipts(
            container,
            tenant_id=layout.tenant_id,
            owners=owners,
        )
        details["memory_consolidations_pre_projection"] = _recover_memory_consolidations(
            container.memory_document_consolidator,
            tenant_id=layout.tenant_id,
            owners=owners,
        )
        details["session_commit_groups"] = _recover_session_commit_groups(
            container.session_commit_service
        )
        details["session_archive_rebuild"] = (
            container.session_commit_service.rebuild_session_archives()
        )
        external_scan: dict[str, Any] = {}
        for owner in owners:
            result = container.memory_document_scanner.scan(
                layout.tenant_id,
                owner,
                force_stable=True,
            )
            if result.deletions_paused:
                raise RuntimeError(f"memory scan deletion reconciliation paused for {owner}: {result.pause_reason}")
            external_scan[owner] = {
                "confirmed": len(result.confirmed_changes),
                "pending": result.pending_change_count,
            }
        details["memory_full_scan"] = external_scan
        rebuild: dict[str, Any] = {}
        for owner in owners:
            rebuild[owner] = container.memory_projection_worker.rebuild_owner(layout.tenant_id, owner)
        details["memory_document_rebuild"] = rebuild
        details["memory_projection_queue"] = _drain_memory_projection_queue(
            container.memory_projection_worker,
            container.queue_store,
        )
        details["memory_consolidations_post_projection"] = _recover_memory_consolidations(
            container.memory_document_consolidator,
            tenant_id=layout.tenant_id,
            owners=owners,
        )
        details["memory_projection_queue_after_consolidation"] = _drain_memory_projection_queue(
            container.memory_projection_worker,
            container.queue_store,
        )
        details["generic_tombstones"] = _drain_tombstones(
            container.tombstone_service,
            tenant_id=layout.tenant_id,
        )
        verified: dict[str, Any] = {}
        for owner in owners:
            verified[owner] = container.memory_projection_worker.verify_owner(layout.tenant_id, owner)
        details["memory_document_verification"] = verified
    except Exception as exc:  # startup is an observable fail-closed boundary.
        container.readiness.transition(
            RuntimeReadinessState.NOT_READY,
            reasons=(f"{type(exc).__name__}: {exc}",),
            details=details,
        )
    else:
        container.readiness.transition(RuntimeReadinessState.READY, details=details)


def _discover_owner_ids(layout: RuntimeLayout, *, limit: int) -> tuple[str, ...]:
    if limit <= 0:
        raise ValueError("memory owner enumeration limit must be positive")
    candidates: set[str] = set()
    source_root = layout.root / "tenants" / layout.tenant_id / "users"
    control_root = layout.tenant_root / "system" / "memory-documents"
    for parent in (source_root, control_root):
        if not parent.exists():
            continue
        if parent.is_symlink() or not parent.is_dir():
            raise RuntimeError("memory owner root is unsafe")
        for child in parent.iterdir():
            if child.name == "sealed-proposals":
                continue
            if child.is_symlink() or not child.is_dir():
                raise RuntimeError("memory owner entry is unsafe")
            candidates.add(MemoryDocumentPathPolicy.trusted_segment(child.name, "owner_user_id"))
            if len(candidates) > limit:
                raise RuntimeError("document owner enumeration exceeded its bound")
    return tuple(sorted(candidates))


def _independent_session_archives(references: tuple[str, ...]) -> tuple[str, ...]:
    archives: set[str] = set()
    for reference in references:
        logical = str(reference).split("#manifest=", 1)[0]
        if logical.startswith("memoryos://user/") and "/sessions/history/" in logical:
            archives.add(logical)
    return tuple(sorted(archives))


def _bounded_owner_ids(layout: RuntimeLayout, tenant_id: str, limit: int) -> tuple[str, ...]:
    if tenant_id != layout.tenant_id:
        raise PermissionError("document owner enumeration crossed the runtime tenant")
    return _discover_owner_ids(layout, limit=limit)


def _recover_memory_consolidations(
    consolidator: MemoryDocumentConsolidator,
    *,
    tenant_id: str,
    owners: tuple[str, ...],
) -> dict[str, Any]:
    per_owner: dict[str, dict[str, object]] = {}
    totals = {
        "examined": 0,
        "completed": 0,
        "awaiting_projection": 0,
        "awaiting_input": 0,
    }
    for owner in owners:
        outcome = consolidator.resume_all(
            tenant_id=tenant_id,
            owner_user_id=owner,
            limit=1_000,
        ).to_dict()
        per_owner[owner] = outcome
        for key in totals:
            value = outcome[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise RuntimeError("consolidation recovery report contains a non-integer count")
            totals[key] += value
    return {**totals, "owners": per_owner}


def _recover_session_commit_groups(service: SessionCommitService) -> dict[str, Any]:
    abandoned = service.commit_group_store.recover_abandoned_leases()
    expired = service.commit_group_store.recover_expired_consumers()
    resumed = 0
    for group in service.resumable_commit_groups(limit=1_000):
        if any(consumer.status == "running" for consumer in group.consumers.values()):
            raise RuntimeError(f"Session commit group has a live lease: {group.group_id}")
        archive = service.archive_store.read_archive_at_manifest(
            group.archive_uri,
            group.manifest_digest,
            tenant_id=group.tenant_id,
        )
        result = service.resume_startup_commit_group(archive, group_id=group.group_id)
        if not result.done or not result.memory_committed:
            raise RuntimeError(f"Session commit group remains incomplete: {group.group_id}")
        resumed += 1
    return {"abandoned_leases": abandoned, "expired_consumers": expired, "resumed": resumed}


def _drain_memory_projection_queue(
    worker: MemoryDocumentProjectionWorker,
    queue_store: QueueStore,
) -> dict[str, int]:
    processed = stale = 0
    for _ in range(1_000):
        stats = queue_store.stats(queue_name=worker.queue_name)
        if not int(stats.get("pending", 0) or 0):
            break
        run = worker.process_pending(limit=100, lease_seconds=300)
        processed += len(run.processed)
        stale += len(run.stale)
    stats = queue_store.stats(queue_name=worker.queue_name)
    if any(int(stats.get(name, 0) or 0) for name in ("pending", "leased", "dead_letter", "quarantine")):
        raise RuntimeError("memory projection queue is not quiescent after startup recovery")
    return {"processed": processed, "stale": stale, **{key: int(value) for key, value in stats.items()}}


def _drain_tombstones(
    service: ProjectionTombstoneService,
    *,
    tenant_id: str,
) -> dict[str, int]:
    processed = stale = 0
    for _ in range(1_000):
        result = service.process_pending(tenant_id=tenant_id, limit=100)
        if result.failed:
            raise RuntimeError("generic projection tombstone replay is incomplete")
        processed += len(result.processed)
        stale += len(result.stale)
        if not result.processed and not result.stale:
            return {"processed": processed, "stale": stale}
    raise RuntimeError("generic projection tombstone replay exceeded its startup bound")


__all__ = ["RuntimeContainer", "build_runtime_container"]
