"""Markdown Memory 领域对象的装配。"""

from __future__ import annotations

from typing import cast

from infrastructure.context.projection.memory_document import MemoryDocumentProjector
from infrastructure.context.retrieval.memory_document_candidates import find_related_memory_documents
from infrastructure.store.contracts.index import MemoryDocumentProjectionStore
from infrastructure.store.contracts.path_lock import PathLock
from infrastructure.store.filesystem.memory_document_store import FileSystemMemoryDocumentStore
from infrastructure.store.memory import (
    MemoryDocumentBootstrapper,
    MemoryDocumentConsolidationStore,
    MemoryDocumentControlStore,
    MemoryDocumentEraseStore,
    MemoryDocumentRevisionStore,
    MemoryDocumentScanner,
    MemoryEditReviewStore,
)
from infrastructure.store.memory.evidence import SealedProposalEraseBackend, SealedProposalStore
from infrastructure.store.trace import RecallTraceEraseBackend
from memory.commit import MemoryDocumentCommitter, MemoryDocumentConsolidator, MemoryDocumentEraser
from memory.commit.evidence.lineage import independent_session_archives
from memory.execute import MemoryDocumentPlanner
from memory.execute.command_service import MemoryCommandService
from memory.execute.external_change import publish_external_change as publish_external_memory_change
from memory.execute.pending_review_service import MemoryEditReviewService
from memory.ports.consolidation import ConsolidationProjectionReader
from memory.worker.document_edit import MemoryDocumentEditWorker
from memory.worker.document_scan import MemoryDocumentScanWorker
from memory.worker.projection.erase_backend import MemoryDocumentCatalogEraseBackend
from memory.worker.projection.worker import MemoryDocumentProjectionWorker
from runtime.config import RuntimeConfig
from runtime.container import MemoryRuntime, StoreRuntime

_MAX_STARTUP_MEMORY_OWNERS = 1_000


def wire_memory(
    stores: StoreRuntime,
    config: RuntimeConfig,
    *,
    readiness,  # noqa: ANN001
    document_store: FileSystemMemoryDocumentStore,
    owner_user_ids,  # noqa: ANN001
) -> MemoryRuntime:
    """连接 Memory Store、Planner、Committer、维护服务和 Worker。"""

    root = config.root_path
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
    planner = MemoryDocumentPlanner(
        document_store,
        max_front_matter_bytes=config.memory_front_matter_max_bytes,
        max_front_matter_depth=config.memory_front_matter_max_depth,
        max_edit_bytes=config.memory_document_max_bytes,
        related_document_finder=lambda tenant, owner, proposal, limit: find_related_memory_documents(
            stores.index,
            tenant_id=tenant,
            owner_user_id=owner,
            proposal=proposal,
            limit=limit,
        ),
        max_related_documents=8,
    )
    erasure_store = MemoryDocumentEraseStore(root)
    committer = MemoryDocumentCommitter(
        document_store,
        control_store,
        revision_store,
        stores.queue,
        erasure_store=erasure_store,
        path_lock=PathLock(stores.lock),
    )
    projector = MemoryDocumentProjector(
        max_front_matter_bytes=config.memory_front_matter_max_bytes,
        max_front_matter_depth=config.memory_front_matter_max_depth,
    )
    projection_worker = MemoryDocumentProjectionWorker(
        document_store,
        control_store,
        cast(MemoryDocumentProjectionStore, stores.index),
        stores.queue,
        projector=projector,
        vector_store=stores.vector,
        embedding_provider=stores.embedding,
        relation_store=stores.relation,
        erasure_store=erasure_store,
    )
    consolidation_store = MemoryDocumentConsolidationStore(root)
    consolidator = MemoryDocumentConsolidator(
        committer,
        cast(ConsolidationProjectionReader, stores.index),
        saga_store=consolidation_store,
    )
    proposal_store = SealedProposalStore(root, tenant_id=config.tenant_id)
    eraser = MemoryDocumentEraser(
        document_store,
        control_store,
        revision_store,
        erase_store=erasure_store,
        review_store=review_store,
        cleanup_backends=(
            MemoryDocumentCatalogEraseBackend(projection_worker),
            SealedProposalEraseBackend(proposal_store),
            RecallTraceEraseBackend(root),
        ),
    )
    command_service = MemoryCommandService(
        planner,
        committer,
        eraser,
        bootstrapper=bootstrapper,
        independent_evidence_locator=lambda tenant, owner, document, _digest: independent_session_archives(
            control_store.lineage_references(tenant, owner, document)
        ),
        readiness=readiness,
        consolidator=consolidator,
        review_store=review_store,
    )
    review_service = MemoryEditReviewService(
        review_store,
        committer,
        erasure_store=erasure_store,
        readiness=readiness,
        consolidator=consolidator,
    )

    def publish_external_change(change) -> None:  # noqa: ANN001
        publish_external_memory_change(
            change,
            committer=committer,
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
    edit_worker = MemoryDocumentEditWorker(
        committer,
        stores.queue,
        readiness=readiness,
    )
    scan_worker = MemoryDocumentScanWorker(
        scanner,
        stores.queue,
        owner_user_ids=owner_user_ids,
        owner_enumeration_limit=_MAX_STARTUP_MEMORY_OWNERS,
        readiness=readiness,
    )
    return MemoryRuntime(
        document_store=document_store,
        control_store=control_store,
        revision_store=revision_store,
        review_store=review_store,
        bootstrapper=bootstrapper,
        planner=planner,
        committer=committer,
        erasure_store=erasure_store,
        consolidation_store=consolidation_store,
        consolidator=consolidator,
        projector=projector,
        scanner=scanner,
        edit_worker=edit_worker,
        scan_worker=scan_worker,
        projection_worker=projection_worker,
        eraser=eraser,
        command_service=command_service,
        review_service=review_service,
        proposal_store=proposal_store,
    )


__all__ = ["wire_memory"]
