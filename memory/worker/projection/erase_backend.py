"""将 Hard Erase 请求落实到 Catalog、Vector 和 Relation 派生层。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from foundation.clock import utc_now
from infrastructure.store.memory.control_store import (
    DocumentDeletionStatus,
    DocumentPublicationBarrier,
)
from memory.ports.erase import DerivedEraseRequest
from memory.worker.projection.model import coerce_persisted_int

if TYPE_CHECKING:
    from memory.worker.projection.worker import MemoryDocumentProjectionWorker


class MemoryDocumentCatalogEraseBackend:
    """清除一个 Memory Document 的全部 Serving 派生数据。"""

    name = "derived.catalog"

    def __init__(self, worker: MemoryDocumentProjectionWorker) -> None:
        self.worker = worker

    def projection_generation_floor(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> int:
        return self.worker._serving_projection_generation_floor(
            tenant_id,
            owner_user_id,
            document_id,
        )

    def erase_document(self, request: DerivedEraseRequest) -> bool:
        barrier = self.worker.control_store.load_publication_barrier(
            request.tenant_id,
            request.owner_user_id,
            request.document_id,
        )
        if barrier is None:
            barrier = self.worker.control_store.write_publication_barrier(
                DocumentPublicationBarrier(
                    tenant_id=request.tenant_id,
                    owner_user_id=request.owner_user_id,
                    document_id=request.document_id,
                    relative_path=request.relative_path,
                    relative_path_digest=request.relative_path_digest,
                    deletion_generation=request.projection_generation_floor + 1,
                    deletion_event_digest=request.erasure_epoch.removeprefix("erase_"),
                    status=DocumentDeletionStatus.HARD_ERASED,
                    updated_at=utc_now(),
                )
            )
        if (
            barrier.status is not DocumentDeletionStatus.HARD_ERASED
            or barrier.deletion_event_digest != request.erasure_epoch.removeprefix("erase_")
        ):
            raise RuntimeError("hard-erasure cleanup is detached from its protected publication barrier")
        existing = self.worker._existing_projection_uris(
            request.tenant_id,
            request.owner_user_id,
            request.document_id,
        )
        current = (
            self.worker._projection_state(
                request.tenant_id,
                request.owner_user_id,
                request.document_id,
            )
            or {}
        )
        if not (
            current.get("deletion_status") == "HARD_ERASED"
            and current.get("deletion_event_digest") == barrier.deletion_event_digest
            and coerce_persisted_int(current.get("deletion_generation") or 0)
            == barrier.deletion_generation
            and not current.get("source_digest")
        ):
            obsolete = self.worker.catalog_store.tombstone_memory_document_projection(
                tenant_id=request.tenant_id,
                owner_user_id=request.owner_user_id,
                document_id=request.document_id,
                deletion_generation=barrier.deletion_generation,
                deletion_event_digest=barrier.deletion_event_digest,
                deletion_status=barrier.status.value,
                relative_path=barrier.relative_path,
            )
            self.worker._remove_obsolete(request.tenant_id, obsolete, existing)
        self.worker._purge_document_derivatives(
            request.tenant_id,
            request.owner_user_id,
            request.document_id,
            request.document_uri,
        )
        self.worker.queue_store.purge_target_jobs(
            queue_name=self.worker.queue_name,
            target_uri=request.document_uri,
            tenant_id=request.tenant_id,
            owner_user_id=request.owner_user_id,
        )
        state = self.worker._projection_state(
            request.tenant_id,
            request.owner_user_id,
            request.document_id,
        )
        return bool(
            state
            and state.get("deletion_status") == "HARD_ERASED"
            and coerce_persisted_int(state.get("deletion_generation") or 0)
            == barrier.deletion_generation
            and not state.get("source_digest")
        )


__all__ = [
    "MemoryDocumentCatalogEraseBackend",
]
