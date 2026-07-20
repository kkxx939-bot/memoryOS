"""记忆文档硬删除事务的只向前协调器。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from functools import partial
from typing import cast

from foundation.clock import utc_now
from infrastructure.store.memory.control_store import (
    DocumentDeletionStatus,
    DocumentIntentStatus,
    DocumentPublicationBarrier,
    MemoryDocumentControlStore,
)
from infrastructure.store.memory.revision_store import MemoryDocumentRevisionStore
from memory.core.model import ABSENT, DocumentEditKind, ManagedDocument, PresentPath, UnsafePath
from memory.core.structure.frontmatter import validate_document_id
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.ports.document_store import MemoryDocumentStore
from memory.ports.erase import (
    _LOCAL_CONTROLS,
    _LOCAL_LIVE,
    _LOCAL_REVIEWS,
    _LOCAL_REVISIONS,
    _MAX_ERASE_RECORDS_PER_OWNER,
    DerivedEraseRequest,
    DocumentEraseCleanupBackend,
    DocumentEraseConflict,
    DocumentEraseFloorProvider,
    DocumentEraseIntegrityError,
    DocumentEraseRecord,
    DocumentEraseRecoveryReport,
    DocumentEraseResult,
    DocumentEraseStatus,
    DocumentEraseStore,
    DocumentReviewPurger,
    _bounded_reference,
    _is_sha256,
    _progress,
    _validate_backend_name,
)


class MemoryDocumentEraser:
    """协调正文、历史版本和派生存储之间可重放的硬删除事务。"""

    def __init__(
        self,
        document_store: MemoryDocumentStore,
        control_store: MemoryDocumentControlStore,
        revision_store: MemoryDocumentRevisionStore,
        *,
        erase_store: DocumentEraseStore,
        review_store: DocumentReviewPurger | None = None,
        cleanup_backends: Sequence[DocumentEraseCleanupBackend] = (),
        clock: Callable[[], str] = utc_now,
    ) -> None:
        self.document_store = document_store
        self.control_store = control_store
        self.revision_store = revision_store
        self.review_store = review_store
        self.cleanup_backends = tuple(cleanup_backends)
        self.clock = clock
        self.erase_store = erase_store
        names = [backend.name for backend in self.cleanup_backends]
        if len(names) != len(set(names)):
            raise ValueError("hard-erasure cleanup backend names must be unique")
        for name in names:
            _validate_backend_name(name)
            if name.startswith("local."):
                raise ValueError("configured cleanup backend cannot use the reserved local namespace")

    def hard_erase(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        expected_source_digest: str,
        relative_path: str = "",
        independent_evidence_retained: Sequence[str] = (),
    ) -> DocumentEraseResult:
        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        identifier = validate_document_id(document_id)
        if not _is_sha256(expected_source_digest):
            raise ValueError("hard erase requires an exact lowercase source digest")
        if relative_path:
            relative_path = MemoryDocumentPathPolicy.normalize_relative_path(relative_path)
        evidence = tuple(sorted({_bounded_reference(item) for item in independent_evidence_retained}))
        backend_names = self._backend_names()
        with self._document_lock(tenant, owner, identifier):
            record = self.erase_store.load(tenant, owner, identifier)
            if record is None:
                path = relative_path or self._registered_path(tenant, owner, identifier)
                if not path:
                    raise DocumentEraseConflict("hard erase requires one exact live registered document")
                live = self.document_store.read_state(tenant, owner, path)
                control = self.control_store.load_control(tenant, owner, identifier)
                if isinstance(live, PresentPath):
                    if live.raw_sha256 != expected_source_digest:
                        raise DocumentEraseConflict("hard erase expected digest does not match the live document")
                elif live == ABSENT:
                    revisions = self.revision_store.list_revisions(tenant, owner, identifier)
                    latest = revisions[-1] if revisions else None
                    if (
                        control is None
                        or control.status != "deleted"
                        or control.relative_path != path
                        or latest is None
                        or latest.state != "ABSENT"
                        or latest.edit_kind is not DocumentEditKind.DELETE
                        or latest.relative_path != path
                        or latest.content_blob_role != "before_delete"
                        or latest.content_blob_digest != expected_source_digest
                    ):
                        raise DocumentEraseConflict("hard erase ABSENT target is not one exact soft-forgotten document")
                else:
                    raise DocumentEraseConflict("hard erase expected digest does not match the live document")
                active = [
                    intent
                    for intent in self.control_store.incomplete_intents(tenant, owner)
                    if intent.document_id == identifier
                    and intent.status not in {DocumentIntentStatus.COMPLETED, DocumentIntentStatus.CONFLICTED}
                ]
                if active:
                    raise DocumentEraseConflict("hard erase requires existing document intents to finish recovery")
                revision_floor = control.logical_revision if control is not None else 0
                projection_floor = max(
                    control.projection_generation if control is not None else 0,
                    self._serving_projection_generation_floor(tenant, owner, identifier),
                    self._protected_projection_generation_floor(tenant, owner, identifier),
                )
                record = self.erase_store.begin(
                    tenant_id=tenant,
                    owner_user_id=owner,
                    document_id=identifier,
                    relative_path=path,
                    source_digest=expected_source_digest,
                    document_revision_floor=revision_floor,
                    projection_generation_floor=projection_floor,
                    backend_names=backend_names,
                    independent_evidence_retained=evidence,
                    started_at=self.clock(),
                )
            elif record.source_digest != expected_source_digest:
                raise DocumentEraseConflict("hard erase retry changed its exact source digest")
            elif evidence and evidence != record.independent_evidence_retained:
                raise DocumentEraseConflict("hard erase retry changed independent evidence disclosure")
            record = self.erase_store.merge_backends(record, backend_names, updated_at=self.clock())
            current_control = self.control_store.load_control(tenant, owner, identifier)
            record = self.erase_store.raise_floors(
                record,
                document_revision_floor=(current_control.logical_revision if current_control is not None else 0),
                projection_generation_floor=max(
                    current_control.projection_generation if current_control is not None else 0,
                    self._serving_projection_generation_floor(tenant, owner, identifier),
                    self._protected_projection_generation_floor(tenant, owner, identifier),
                ),
                updated_at=self.clock(),
            )
            self._seal_hard_publication_barrier(record)
            if record.status != DocumentEraseStatus.ERASED:
                record = self._run(record, relative_path=relative_path)
            return DocumentEraseResult(record, record.independent_evidence_retained)

    def recover_owner(
        self,
        tenant_id: str,
        owner_user_id: str,
        *,
        limit: int = _MAX_ERASE_RECORDS_PER_OWNER,
    ) -> DocumentEraseRecoveryReport:
        """在数量上限内重放所有配置的清理后端。"""

        completed: list[str] = []
        pending: list[str] = []
        for record in self.erase_store.records(tenant_id, owner_user_id, limit=limit):
            result = self.hard_erase(
                tenant_id=record.tenant_id,
                owner_user_id=record.owner_user_id,
                document_id=record.document_id,
                expected_source_digest=record.source_digest,
                relative_path=record.relative_path,
            )
            (completed if result.completed else pending).append(record.document_id)
        return DocumentEraseRecoveryReport(tuple(completed), tuple(pending))

    def _seal_hard_publication_barrier(
        self,
        record: DocumentEraseRecord,
    ) -> DocumentPublicationBarrier:
        current = self.control_store.load_publication_barrier(
            record.tenant_id,
            record.owner_user_id,
            record.document_id,
        )
        digest = record.erasure_epoch.removeprefix("erase_")
        generation = record.projection_generation_floor + 1
        if current is not None:
            if current.status is DocumentDeletionStatus.HARD_ERASED:
                if (
                    current.deletion_event_digest != digest
                    or current.relative_path_digest != record.relative_path_digest
                ):
                    raise DocumentEraseIntegrityError(
                        "hard-erasure epoch conflicts with the protected publication barrier"
                    )
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
                updated_at=self.clock(),
            )
        )

    def _serving_projection_generation_floor(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> int:
        floor = 0
        for backend in self.cleanup_backends:
            provider = getattr(backend, "projection_generation_floor", None)
            if not callable(provider):
                continue
            value = cast(DocumentEraseFloorProvider, backend).projection_generation_floor(
                tenant_id,
                owner_user_id,
                document_id,
            )
            if int(value) < 0:
                raise DocumentEraseIntegrityError("hard-erasure serving generation floor is negative")
            floor = max(floor, int(value))
        return floor

    def _protected_projection_generation_floor(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> int:
        barrier = self.control_store.load_publication_barrier(
            tenant_id,
            owner_user_id,
            document_id,
        )
        if barrier is None:
            return 0
        if barrier.status is DocumentDeletionStatus.HARD_ERASED:
            return max(0, barrier.deletion_generation - 1)
        return barrier.deletion_generation

    def _run(self, record: DocumentEraseRecord, *, relative_path: str) -> DocumentEraseRecord:
        local_actions: tuple[tuple[str, Callable[[], bool]], ...] = (
            (_LOCAL_LIVE, lambda: self._erase_live(record, relative_path)),
            (_LOCAL_REVISIONS, lambda: self._erase_revisions(record)),
            *(((_LOCAL_REVIEWS, lambda: self._erase_reviews(record)),) if self.review_store is not None else ()),
            (_LOCAL_CONTROLS, lambda: self._erase_controls(record)),
        )
        for name, action in local_actions:
            if _progress(record, name).acknowledged:
                continue
            record = self._attempt(record, name, action)
            if not _progress(record, name).acknowledged:
                return self.erase_store.finish(record, completed_at=self.clock())

        request = DerivedEraseRequest(
            tenant_id=record.tenant_id,
            owner_user_id=record.owner_user_id,
            document_id=record.document_id,
            document_uri=record.document_uri,
            relative_path=record.relative_path,
            document_kind=record.document_kind,
            erasure_epoch=record.erasure_epoch,
            source_digest=record.source_digest,
            document_revision_floor=record.document_revision_floor,
            projection_generation_floor=record.projection_generation_floor,
            relative_path_digest=record.relative_path_digest,
        )
        for backend in self.cleanup_backends:
            if _progress(record, backend.name).acknowledged:
                continue
            record = self._attempt(
                record,
                backend.name,
                partial(backend.erase_document, request),
            )
        if not record.pending_backends:
            self.control_store.scrub_hard_erasure_path(
                record.tenant_id,
                record.owner_user_id,
                record.document_id,
                expected_relative_path_digest=record.relative_path_digest,
                expected_deletion_event_digest=record.erasure_epoch.removeprefix("erase_"),
                updated_at=self.clock(),
            )
        return self.erase_store.finish(record, completed_at=self.clock())

    def _attempt(
        self,
        record: DocumentEraseRecord,
        backend_name: str,
        action: Callable[[], bool],
    ) -> DocumentEraseRecord:
        attempted_at = self.clock()
        try:
            acknowledged = bool(action())
            failure_code = "" if acknowledged else "NOT_ACKNOWLEDGED"
        except Exception as exc:  # noqa: BLE001 - 耐久 Saga 会记录不含正文的类型化失败。
            acknowledged = False
            failure_code = type(exc).__name__
        return self.erase_store.record_attempt(
            record,
            backend_name,
            acknowledged=acknowledged,
            attempted_at=attempted_at,
            failure_code=failure_code,
        )

    def _erase_live(self, record: DocumentEraseRecord, relative_path: str) -> bool:
        path = relative_path or self._registered_path(
            record.tenant_id,
            record.owner_user_id,
            record.document_id,
        )
        if not path:
            return True
        live = self.document_store.read_state(record.tenant_id, record.owner_user_id, path)
        if live == ABSENT:
            return True
        if isinstance(live, UnsafePath):
            raise DocumentEraseConflict("live document path became unsafe during erasure")
        if not isinstance(live, PresentPath) or live.raw_sha256 != record.source_digest:
            raise DocumentEraseConflict("live document changed after the durable erasure intent")
        self.document_store.full_scan(record.tenant_id, record.owner_user_id)
        self.document_store.delete(
            record.tenant_id,
            record.owner_user_id,
            record.document_id,
            expected_state=live,
        )
        return self.document_store.read_state(record.tenant_id, record.owner_user_id, path) == ABSENT

    def _erase_revisions(self, record: DocumentEraseRecord) -> bool:
        self.revision_store.purge_document(record.tenant_id, record.owner_user_id, record.document_id)
        return True

    def _erase_reviews(self, record: DocumentEraseRecord) -> bool:
        assert self.review_store is not None
        self.review_store.purge_document(record.tenant_id, record.owner_user_id, record.document_id)
        return True

    def _erase_controls(self, record: DocumentEraseRecord) -> bool:
        self.control_store.purge_document(record.tenant_id, record.owner_user_id, record.document_id)
        return True

    def _registered_path(self, tenant_id: str, owner_user_id: str, document_id: str) -> str:
        scan = self.document_store.full_scan(tenant_id, owner_user_id)
        if not scan.complete or scan.errors:
            raise DocumentEraseConflict("hard erase requires a complete live document registration scan")
        matches = [
            item.relative_path
            for item in scan.registrations
            if isinstance(item, ManagedDocument) and item.document_id == document_id
        ]
        if len(matches) > 1:
            raise DocumentEraseConflict("hard erase document identity is duplicated")
        return matches[0] if matches else ""

    def _backend_names(self) -> tuple[str, ...]:
        local = [_LOCAL_LIVE, _LOCAL_REVISIONS]
        if self.review_store is not None:
            local.append(_LOCAL_REVIEWS)
        local.append(_LOCAL_CONTROLS)
        return tuple(local + [backend.name for backend in self.cleanup_backends])

    def _document_lock(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> AbstractContextManager[None]:
        return self.erase_store.document_lock(tenant_id, owner_user_id, document_id)


__all__ = ["MemoryDocumentEraser"]
