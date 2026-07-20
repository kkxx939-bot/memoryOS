"""维护 Markdown Memory 的删除屏障和派生层清理。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from foundation.clock import utc_now
from infrastructure.store.memory.control_store import (
    DocumentControlRecord,
    DocumentDeletionStatus,
    DocumentPublicationBarrier,
)
from memory.core.model import ManagedDocument
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.ports.erase import DocumentEraseRecord
from memory.worker.projection.model import coerce_persisted_int


class ProjectionErasureMixin:
    """把耐久删除事实同步为 Serving 层屏障。"""

    def _fence_durable_erasure(
        self: Any,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> DocumentPublicationBarrier | None:
        """把一个耐久擦除 epoch 重放到受保护屏障和 Serving 删除屏障。"""

        record = self.erasure_store.load(tenant_id, owner_user_id, document_id)
        if record is None:
            return None
        barrier = self._ensure_hard_erasure_barrier(record)
        self._mirror_barrier(barrier)
        return barrier

    def _ensure_hard_erasure_barrier(
        self: Any,
        record: DocumentEraseRecord,
    ) -> DocumentPublicationBarrier:
        current = self.control_store.load_publication_barrier(
            record.tenant_id,
            record.owner_user_id,
            record.document_id,
        )
        state = (
            self._projection_state(
                record.tenant_id,
                record.owner_user_id,
                record.document_id,
            )
            or {}
        )
        control = self.control_store.load_control(
            record.tenant_id,
            record.owner_user_id,
            record.document_id,
        )
        live_floor = max(
            record.projection_generation_floor,
            control.projection_generation if control is not None else 0,
            self._serving_projection_generation_floor(
                record.tenant_id,
                record.owner_user_id,
                record.document_id,
            ),
        )
        generation = max(
            live_floor + 1,
            coerce_persisted_int(state.get("deletion_generation") or 0),
        )
        digest = record.erasure_epoch.removeprefix("erase_")
        if current is not None:
            if current.status is DocumentDeletionStatus.HARD_ERASED:
                if (
                    current.deletion_event_digest != digest
                    or current.relative_path_digest != record.relative_path_digest
                ):
                    raise RuntimeError("durable erasure conflicts with its hard publication barrier")
                generation = max(generation, current.deletion_generation)
            else:
                generation = max(generation, current.deletion_generation + 1)
        return self.control_store.write_publication_barrier(
            DocumentPublicationBarrier(
                tenant_id=record.tenant_id,
                owner_user_id=record.owner_user_id,
                document_id=record.document_id,
                relative_path=record.relative_path,
                relative_path_digest=record.relative_path_digest,
                deletion_generation=generation,
                deletion_event_digest=digest,
                status=DocumentDeletionStatus.HARD_ERASED,
                updated_at=utc_now(),
            )
        )

    def _serving_projection_generation_floor(
        self: Any,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> int:
        state = self._projection_state(tenant_id, owner_user_id, document_id) or {}
        floor = coerce_persisted_int(state.get("projection_generation") or 0)
        for record in self._owner_document_records(tenant_id, owner_user_id):
            if record.document_id == document_id:
                floor = max(floor, int(record.projection_generation))
        return floor

    def _purge_document_derivatives(
        self: Any,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        document_uri: str,
    ) -> None:
        """Catalog 写入 tombstone 或崩溃恢复后，按精确身份重放清理。"""

        if self.vector_store is not None:
            deleted = self.vector_store.delete_by_filter(
                {
                    "tenant_id": tenant_id,
                    "owner_user_id": owner_user_id,
                    "document_id": document_id,
                }
            )
            if not isinstance(deleted, int) or isinstance(deleted, bool) or deleted < 0:
                raise TypeError("vector delete-by-filter returned an invalid deletion count")
        if self.relation_store is not None:
            with self.erasure_store.owner_relation_lock(tenant_id, owner_user_id):
                while self.relation_store.delete_memory_document_relations(
                    document_uri,
                    tenant_id=tenant_id,
                    owner_user_id=owner_user_id,
                    limit=1_000,
                ):
                    pass

    def _projection_state(
        self: Any,
        tenant: str,
        owner: str,
        document_id: str,
    ) -> Mapping[str, object] | None:
        return self.catalog_store.get_memory_document_projection_state(
            tenant_id=tenant,
            owner_user_id=owner,
            document_id=document_id,
        )

    @staticmethod
    def _control_authorizes_restored_registration(
        control: DocumentControlRecord | None,
        barrier: DocumentPublicationBarrier,
        registration: ManagedDocument,
    ) -> bool:
        return bool(
            barrier.status is DocumentDeletionStatus.SOFT_FORGOTTEN
            and control is not None
            and control.status == "present"
            and control.document_id == registration.document_id
            and control.relative_path == registration.relative_path
            and control.raw_sha256 == registration.raw_sha256
            and control.restored_from_deletion_generation == barrier.deletion_generation
            and control.projection_generation > barrier.deletion_generation
        )

    def _mirror_barrier(self: Any, barrier: DocumentPublicationBarrier) -> tuple[str, ...]:
        """根据受保护控制状态重新创建或刷新 SQLite tombstone。"""

        state = self._projection_state(
            barrier.tenant_id,
            barrier.owner_user_id,
            barrier.document_id,
        )
        if state is not None:
            state_status = str(state.get("deletion_status") or "")
            state_deletion_generation = coerce_persisted_int(state.get("deletion_generation") or 0)
            state_digest = str(state.get("deletion_event_digest") or "")
            if (
                state_status == barrier.status.value
                and state_deletion_generation == barrier.deletion_generation
                and state_digest == barrier.deletion_event_digest
                and not str(state.get("source_digest") or "")
            ):
                self._purge_document_derivatives(
                    barrier.tenant_id,
                    barrier.owner_user_id,
                    barrier.document_id,
                    MemoryDocumentPathPolicy.document_uri(
                        barrier.owner_user_id,
                        barrier.document_id,
                    ),
                )
                return ()
            current_generation = coerce_persisted_int(state.get("projection_generation") or 0)
            if current_generation > barrier.deletion_generation and not state_status:
                control = self.control_store.load_control(
                    barrier.tenant_id,
                    barrier.owner_user_id,
                    barrier.document_id,
                )
                if self._control_authorizes_restored_serving(control, barrier, state):
                    return ()
                raise RuntimeError("deletion barrier is older than an unauthorized live serving publication")
        obsolete = self.catalog_store.tombstone_memory_document_projection(
            tenant_id=barrier.tenant_id,
            owner_user_id=barrier.owner_user_id,
            document_id=barrier.document_id,
            deletion_generation=barrier.deletion_generation,
            deletion_event_digest=barrier.deletion_event_digest,
            deletion_status=barrier.status.value,
            relative_path=barrier.relative_path,
        )
        self._purge_document_derivatives(
            barrier.tenant_id,
            barrier.owner_user_id,
            barrier.document_id,
            MemoryDocumentPathPolicy.document_uri(
                barrier.owner_user_id,
                barrier.document_id,
            ),
        )
        return obsolete

    @staticmethod
    def _control_authorizes_restored_serving(
        control: DocumentControlRecord | None,
        barrier: DocumentPublicationBarrier,
        state: Mapping[str, object],
    ) -> bool:
        return bool(
            barrier.status is DocumentDeletionStatus.SOFT_FORGOTTEN
            and control is not None
            and control.status == "present"
            and control.document_id == barrier.document_id
            and control.restored_from_deletion_generation == barrier.deletion_generation
            and control.projection_generation > barrier.deletion_generation
            and coerce_persisted_int(state.get("projection_generation") or 0)
            == control.projection_generation
            and str(state.get("source_digest") or "") == control.raw_sha256
            and str(state.get("relative_path") or "") == control.relative_path
            and not str(state.get("deletion_status") or "")
        )


__all__ = ["ProjectionErasureMixin"]
