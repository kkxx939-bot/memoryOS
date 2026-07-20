"""发布单个 live Markdown 文档及其链接投影。"""

from __future__ import annotations

import hashlib
from typing import Any

from infrastructure.store.memory.control_store import DocumentDeletionStatus
from infrastructure.store.model.catalog import CatalogRecord
from memory.core.model import ManagedDocument, PresentPath
from memory.ports.document_store import DocumentConflictError
from memory.ports.erase import DocumentErasedError
from memory.worker.projection.model import coerce_persisted_int


class ProjectionPublicationMixin:
    """封装精确读取、Catalog 发布和链接刷新流程。"""

    def _publish_live(
        self: Any,
        *,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        relative_path: str,
        source_digest: str,
        document_revision: int,
        projection_generation: int,
        expected_previous_generation: int | None = None,
        restored_from_deletion_generation: int = 0,
    ) -> tuple[str, ...]:
        with self.erasure_store.document_lock(tenant_id, owner_user_id, document_id):
            return self._publish_live_locked(
                tenant_id=tenant_id,
                owner_user_id=owner_user_id,
                document_id=document_id,
                relative_path=relative_path,
                source_digest=source_digest,
                document_revision=document_revision,
                projection_generation=projection_generation,
                expected_previous_generation=expected_previous_generation,
                restored_from_deletion_generation=restored_from_deletion_generation,
            )

    def _publish_live_locked(
        self: Any,
        *,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        relative_path: str,
        source_digest: str,
        document_revision: int,
        projection_generation: int,
        expected_previous_generation: int | None,
        restored_from_deletion_generation: int,
    ) -> tuple[str, ...]:
        erasure_barrier = self._fence_durable_erasure(tenant_id, owner_user_id, document_id)
        if erasure_barrier is not None:
            raise DocumentErasedError("durable hard-erasure epoch blocks live projection")
        barrier = self.control_store.load_publication_barrier(
            tenant_id,
            owner_user_id,
            document_id,
        )
        if barrier is not None:
            if barrier.status is DocumentDeletionStatus.HARD_ERASED:
                raise DocumentErasedError("hard-erased document identity cannot be projected")
            if (
                restored_from_deletion_generation != barrier.deletion_generation
                or projection_generation <= barrier.deletion_generation
            ):
                raise DocumentConflictError("live projection is blocked by its durable deletion barrier")
        elif restored_from_deletion_generation:
            raise DocumentConflictError("live projection claims restore without a durable deletion barrier")
        state = self.document_store.read_state(tenant_id, owner_user_id, relative_path)
        if not isinstance(state, PresentPath) or state.raw_sha256 != source_digest:
            raise DocumentConflictError("live Markdown changed after its projection event")
        raw = self.document_store.read_raw(tenant_id, owner_user_id, document_id=document_id)
        if hashlib.sha256(raw).hexdigest() != source_digest:
            raise DocumentConflictError("live Markdown changed during projection")
        projection = self.projector.project(
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            relative_path=relative_path,
            raw_bytes=raw,
            source_digest=source_digest,
            document_revision=document_revision,
            projection_generation=projection_generation,
        )
        if projection.document_id != document_id:
            raise DocumentConflictError("live Markdown identity differs from the projection event")
        document_record, block_records = self._records(projection)
        vector_rows = self._prepare_vector_rows(block_records)
        previous = expected_previous_generation
        if previous is None:
            current = self._projection_state(tenant_id, owner_user_id, document_id)
            previous = coerce_persisted_int((current or {}).get("projection_generation") or 0)
        serving_state = self._projection_state(tenant_id, owner_user_id, document_id)
        restore_soft_deleted = bool(
            barrier is not None and str((serving_state or {}).get("deletion_status") or "") == "SOFT_FORGOTTEN"
        )
        existing_uris = self._existing_projection_uris(tenant_id, owner_user_id, document_id)
        inserted_vectors: list[str] = []
        try:
            if self.vector_store is not None:
                for row_id, embedding, metadata in vector_rows:
                    self.vector_store.upsert_vector(row_id, embedding, metadata)
                    inserted_vectors.append(row_id)
            obsolete = self.catalog_store.replace_memory_document_projection(
                document_record,
                block_records,
                previous,
                tenant_id=tenant_id,
                owner_user_id=owner_user_id,
                restore_soft_deleted=restore_soft_deleted,
            )
        except Exception:
            if self.vector_store is not None:
                for row_id in inserted_vectors:
                    self.vector_store.delete_vector(row_id)
            raise
        self._remove_obsolete(tenant_id, obsolete, existing_uris)
        self._replace_document_links(document_record, raw)
        return obsolete

    def _refresh_live_document_links(
        self: Any,
        tenant_id: str,
        owner_user_id: str,
        registration: ManagedDocument,
        document_record: CatalogRecord,
    ) -> None:
        """仅在精确 live 投影仍受屏障保护时替换链接边。"""

        with self.erasure_store.document_lock(tenant_id, owner_user_id, registration.document_id):
            if self._fence_durable_erasure(tenant_id, owner_user_id, registration.document_id) is not None:
                return
            barrier = self.control_store.load_publication_barrier(
                tenant_id,
                owner_user_id,
                registration.document_id,
            )
            if barrier is not None and barrier.status is DocumentDeletionStatus.HARD_ERASED:
                self._mirror_barrier(barrier)
                return
            state = self._projection_state(tenant_id, owner_user_id, registration.document_id) or {}
            if (
                str(state.get("deletion_status") or "")
                or coerce_persisted_int(state.get("projection_generation") or 0)
                != document_record.projection_generation
                or str(state.get("source_digest") or "") != registration.raw_sha256
            ):
                return
            raw = self.document_store.read_raw(
                tenant_id,
                owner_user_id,
                relative_path=registration.relative_path,
            )
            if hashlib.sha256(raw).hexdigest() != registration.raw_sha256:
                raise DocumentConflictError("live Markdown changed during link projection")
            self._replace_document_links(document_record, raw)


__all__ = ["ProjectionPublicationMixin"]
