"""Projection worker for rebuildable Markdown-memory serving rows.

The queue payload is deliberately content-free.  Every run re-reads the
trusted control record and the exact live Markdown bytes before publication.
"""

from __future__ import annotations

import hashlib
import math
import posixpath
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, SupportsIndex, SupportsInt, cast

from memoryos.contextdb.catalog import (
    CatalogProjectionStatus,
    CatalogRecord,
    CatalogRecordKind,
    catalog_vector_metadata,
)
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.embedding import EmbeddingProvider
from memoryos.contextdb.store.index_store import MemoryDocumentProjectionStore
from memoryos.contextdb.store.queue_store import QueueJob, QueueStore
from memoryos.contextdb.store.relation_store import RelationStore
from memoryos.contextdb.store.vector import VectorStore, vector_row_id
from memoryos.core.clock import utc_now
from memoryos.memory.documents.control_store import (
    DocumentControlRecord,
    DocumentDeletionStatus,
    DocumentPublicationBarrier,
    MemoryDocumentControlStore,
    deletion_event_digest,
)
from memoryos.memory.documents.erase import (
    DerivedEraseRequest,
    DocumentErasedError,
    DocumentEraseRecord,
    MemoryDocumentEraseStore,
)
from memoryos.memory.documents.model import DocumentEditKind, ManagedDocument, PresentPath
from memoryos.memory.documents.path_policy import MemoryDocumentPathPolicy
from memoryos.memory.documents.projection import MemoryDocumentProjection, MemoryDocumentProjector
from memoryos.memory.documents.store import DocumentConflictError, MemoryDocumentStore

_MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]\n]{1,500}\]\(([^)\n]{1,2000})\)")


_IntConvertible = str | bytes | bytearray | SupportsInt | SupportsIndex


class _CatalogLister(Protocol):
    def __call__(
        self,
        *,
        tenant_id: str,
        filters: Mapping[str, object],
        limit: int,
    ) -> Sequence[CatalogRecord]: ...


class _CatalogBatchScanner(Protocol):
    def __call__(
        self,
        *,
        tenant_id: str,
        after_record_key: str,
        filters: Mapping[str, object],
        limit: int,
    ) -> Sequence[CatalogRecord]: ...


def _coerce_int(value: object) -> int:
    """Apply ``int`` to one persisted scalar without changing its semantics."""

    return int(cast(_IntConvertible, value))


@dataclass(frozen=True)
class MemoryProjectionRun:
    processed: tuple[str, ...] = ()
    stale: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()


class MemoryDocumentProjectionWorker:
    """Consume document change events and atomically replace Catalog rows."""

    queue_name = "memory_projection"

    def __init__(
        self,
        document_store: MemoryDocumentStore,
        control_store: MemoryDocumentControlStore,
        catalog_store: MemoryDocumentProjectionStore,
        queue_store: QueueStore,
        *,
        projector: MemoryDocumentProjector | None = None,
        vector_store: VectorStore | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        relation_store: RelationStore | None = None,
        erasure_store: MemoryDocumentEraseStore | None = None,
        lease_owner: str = "memory-document-projector",
    ) -> None:
        self.document_store = document_store
        self.control_store = control_store
        self.catalog_store = catalog_store
        self.queue_store = queue_store
        self.projector = projector or MemoryDocumentProjector()
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.relation_store = relation_store
        self.erasure_store = erasure_store or MemoryDocumentEraseStore(control_store.root)
        self.lease_owner = lease_owner

    def process_pending(self, *, limit: int = 10, lease_seconds: int = 60) -> MemoryProjectionRun:
        jobs = self.queue_store.lease(
            self.queue_name,
            lease_owner=self.lease_owner,
            limit=limit,
            lease_seconds=lease_seconds,
        )
        processed: list[str] = []
        stale: list[str] = []
        failed: list[str] = []
        for job in jobs:
            try:
                outcome = self.process_job(job)
                self.queue_store.ack(job)
                (stale if outcome == "stale" else processed).append(job.job_id)
            except (DocumentErasedError, DocumentConflictError, ValueError, RuntimeError, OSError) as exc:
                failed.append(job.job_id)
                self.queue_store.retry(
                    job,
                    type(exc).__name__,
                    max_retries=5,
                    retryable=not isinstance(exc, (DocumentErasedError, ValueError)),
                )
        return MemoryProjectionRun(tuple(processed), tuple(stale), tuple(failed))

    def process_job(self, job: QueueJob) -> str:
        payload = self._payload(job)
        tenant = str(payload["tenant_id"])
        owner = str(payload["owner_user_id"])
        document_id = str(payload["document_id"])
        generation = _coerce_int(payload["projection_generation"])
        edit_kind = DocumentEditKind(str(payload["edit_kind"]))
        barrier = self.control_store.load_publication_barrier(tenant, owner, document_id)
        if barrier is not None and barrier.status is DocumentDeletionStatus.HARD_ERASED:
            with self.erasure_store.document_lock(tenant, owner, document_id):
                durable = self._fence_durable_erasure(tenant, owner, document_id)
                if durable is None:
                    self._mirror_barrier(barrier)
            return "stale"
        self.erasure_store.assert_projection_allowed(
            tenant,
            owner,
            document_id,
            projection_generation=generation,
        )
        serving_state = self._projection_state(tenant, owner, document_id)
        control = self.control_store.load_control(tenant, owner, document_id)
        if control is None:
            raise RuntimeError("projection event has no durable document control record")
        if generation < control.projection_generation:
            return "stale"
        if generation > control.projection_generation:
            raise RuntimeError("projection event is newer than its durable control record")
        if str(payload["event_id"]) != control.last_event_id:
            raise ValueError("projection event identity differs from durable control")
        expected_after = str(payload["after_raw_digest"])
        if barrier is None and serving_state and str(serving_state.get("deletion_status") or ""):
            raise RuntimeError("serving deletion state has no protected control-store barrier")
        if barrier is not None and generation <= barrier.deletion_generation:
            if (
                edit_kind is DocumentEditKind.DELETE
                and generation == barrier.deletion_generation
                and control.status == "deleted"
                and not expected_after
                and self._deletion_digest(payload) == barrier.deletion_event_digest
            ):
                self._mirror_barrier(barrier)
                return "processed"
            return "stale"
        if control.status == "present":
            if expected_after != control.raw_sha256:
                raise ValueError("projection event digest differs from durable control")
            live_state = self.document_store.read_state(
                tenant,
                owner,
                control.relative_path,
            )
            if (
                not isinstance(live_state, PresentPath)
                or live_state.raw_sha256 != control.raw_sha256
            ):
                # The projection event is no longer an exact source fact.  A
                # missing/changed file is reconciled only by the stability
                # scanner; this rebuildable job can be acknowledged as stale
                # so startup does not manufacture deletion authority or loop.
                return "stale"
            restored_generation = 0
            if barrier is not None:
                if (
                    barrier.status is not DocumentDeletionStatus.SOFT_FORGOTTEN
                    or control.restored_from_deletion_generation != barrier.deletion_generation
                    or control.projection_generation <= barrier.deletion_generation
                ):
                    raise RuntimeError("live control is not authorized past its deletion barrier")
                restored_generation = barrier.deletion_generation
                if serving_state is None or str(serving_state.get("deletion_status") or ""):
                    self._mirror_barrier(barrier)
            self._publish_live(
                tenant_id=tenant,
                owner_user_id=owner,
                document_id=document_id,
                relative_path=control.relative_path,
                source_digest=control.raw_sha256,
                document_revision=control.logical_revision,
                projection_generation=generation,
                restored_from_deletion_generation=restored_generation,
            )
        else:
            if expected_after:
                raise ValueError("deleted projection event cannot claim live source bytes")
            if barrier is None:
                raise RuntimeError("deleted projection event has no protected publication barrier")
            self._mirror_barrier(barrier)
        return "processed"

    def rebuild_owner(self, tenant_id: str, owner_user_id: str) -> dict[str, int]:
        """Reconcile one bounded user tree from live source without trusting a watcher."""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        scan = self.document_store.full_scan(tenant, owner)
        if not scan.complete or scan.errors or scan.unsafe_paths:
            raise RuntimeError("memory document rebuild requires one complete safe full scan")
        projected = 0
        skipped = 0
        deleted = 0
        pending_missing = 0
        live_document_ids = {item.document_id for item in scan.managed}
        barriers = {
            barrier.document_id: barrier
            for barrier in self.control_store.publication_barriers(tenant, owner)
        }
        for document_id, durable_barrier in barriers.items():
            if document_id not in live_document_ids:
                with self.erasure_store.document_lock(tenant, owner, document_id):
                    durable = self._fence_durable_erasure(tenant, owner, document_id)
                    if durable is None:
                        self._mirror_barrier(durable_barrier)
                control = self.control_store.load_control(tenant, owner, document_id)
                restored_after_barrier = bool(
                    durable_barrier.status is DocumentDeletionStatus.SOFT_FORGOTTEN
                    and control is not None
                    and control.status == "present"
                    and control.restored_from_deletion_generation
                    == durable_barrier.deletion_generation
                    and control.projection_generation > durable_barrier.deletion_generation
                )
                deleted += int(not restored_after_barrier)
        for registration in scan.registrations:
            if not isinstance(registration, ManagedDocument):
                raise RuntimeError("memory document rebuild found an unmanaged or duplicate identity")
            with self.erasure_store.document_lock(tenant, owner, registration.document_id):
                erasure_barrier = self._fence_durable_erasure(
                    tenant,
                    owner,
                    registration.document_id,
                )
            if erasure_barrier is not None:
                barriers[registration.document_id] = erasure_barrier
                skipped += 1
                continue
            control = self.control_store.load_control(tenant, owner, registration.document_id)
            barrier = barriers.get(registration.document_id)
            state = self._projection_state(tenant, owner, registration.document_id)
            if barrier is None and state is not None and str(state.get("deletion_status") or ""):
                # A concurrent scanner/eraser can publish the protected
                # barrier after this rebuild took its initial owner snapshot.
                # Re-read durable authority before classifying the serving
                # tombstone as detached.
                barrier = self.control_store.load_publication_barrier(
                    tenant,
                    owner,
                    registration.document_id,
                )
                if barrier is not None:
                    barriers[registration.document_id] = barrier
            restored_from_deletion_generation = 0
            if barrier is not None:
                if not self._control_authorizes_restored_registration(
                    control,
                    barrier,
                    registration,
                ):
                    self._mirror_barrier(barrier)
                    skipped += 1
                    continue
                restored_from_deletion_generation = barrier.deletion_generation
                if state is None or str(state.get("deletion_status") or ""):
                    self._mirror_barrier(barrier)
                    state = self._projection_state(tenant, owner, registration.document_id)
            elif state is not None and str(state.get("deletion_status") or ""):
                raise RuntimeError("serving deletion state has no protected control-store barrier")
            deletion_status = str((state or {}).get("deletion_status") or "")
            if deletion_status and not restored_from_deletion_generation:
                skipped += 1
                continue
            current_generation = _coerce_int((state or {}).get("projection_generation") or 0)
            current_digest = str((state or {}).get("source_digest") or "")
            current_path = str((state or {}).get("relative_path") or "")
            if current_digest == registration.raw_sha256 and current_path == registration.relative_path:
                skipped += 1
                continue
            generation = current_generation + 1
            if (
                control is not None
                and control.status == "present"
                and control.relative_path == registration.relative_path
                and control.raw_sha256 == registration.raw_sha256
            ):
                generation = max(generation, control.projection_generation)
            revision = int(control.logical_revision if control is not None else generation)
            self._publish_live(
                tenant_id=tenant,
                owner_user_id=owner,
                document_id=registration.document_id,
                relative_path=registration.relative_path,
                source_digest=registration.raw_sha256,
                document_revision=max(1, revision),
                projection_generation=generation,
                expected_previous_generation=current_generation,
                restored_from_deletion_generation=restored_from_deletion_generation,
            )
            projected += 1
        # Link targets may sort after their sources.  Rebuild document links
        # only after every live document row has been published.
        records_by_id = {
            record.document_id: record for record in self._owner_document_records(tenant, owner)
        }
        for registration in scan.managed:
            record = records_by_id.get(registration.document_id)
            if record is None:
                continue
            self._refresh_live_document_links(tenant, owner, registration, record)
        for record in self._owner_document_records(tenant, owner):
            if record.document_id in live_document_ids:
                continue
            state = self._projection_state(tenant, owner, record.document_id) or {}
            if str(state.get("deletion_status") or ""):
                continue
            # A single full-scan absence is not deletion authority.  Only the
            # stability scanner/committer may persist the deletion event and
            # publication barrier that authorizes a serving tombstone.
            pending_missing += 1
        result = {
            "projected": projected,
            "skipped": skipped,
            "deleted": deleted,
            "documents": len(scan.managed),
        }
        if pending_missing:
            result["pending_missing"] = pending_missing
        return result

    def verify_owner(self, tenant_id: str, owner_user_id: str) -> dict[str, int]:
        """Prove every live registration has one matching serving generation."""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        scan = self.document_store.full_scan(tenant, owner)
        if not scan.complete or scan.errors or scan.unsafe_paths:
            raise RuntimeError("memory document verification requires one complete safe full scan")
        projected = {record.document_id: record for record in self._owner_document_records(tenant, owner)}
        live = {item.document_id: item for item in scan.managed}
        for document_id, registration in live.items():
            state = self._projection_state(tenant, owner, document_id)
            if state is None or str(state.get("deletion_status") or ""):
                raise RuntimeError("live Markdown is blocked or missing from serving projection state")
            record = projected.get(document_id)
            if (
                record is None
                or record.source_digest != registration.raw_sha256
                or str(record.metadata.get("relative_path") or "") != registration.relative_path
                or record.projection_generation
                != _coerce_int(state.get("projection_generation") or 0)
            ):
                raise RuntimeError("live Markdown and serving Catalog projection disagree")
        stale = set(projected) - set(live)
        # Missing live bytes stay out of retrieval through source hydration,
        # but a stability-window absence must not make startup NOT_READY or
        # synthesize deletion authority in the rebuild path.
        result = {
            "verified": len(live),
            "projected": len(projected),
        }
        if stale:
            result["pending_missing"] = len(stale)
            result["degraded"] = len(stale)
        return result

    def _publish_live(
        self,
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
        self,
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
            previous = _coerce_int((current or {}).get("projection_generation") or 0)
        serving_state = self._projection_state(tenant_id, owner_user_id, document_id)
        restore_soft_deleted = bool(
            barrier is not None
            and str((serving_state or {}).get("deletion_status") or "") == "SOFT_FORGOTTEN"
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
        self,
        tenant_id: str,
        owner_user_id: str,
        registration: ManagedDocument,
        document_record: CatalogRecord,
    ) -> None:
        """Replace link edges only while the exact live publication remains fenced."""

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
                or _coerce_int(state.get("projection_generation") or 0)
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

    def _fence_durable_erasure(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> DocumentPublicationBarrier | None:
        """Replay an erasure epoch into the protected and serving deletion fences."""

        record = self.erasure_store.load(tenant_id, owner_user_id, document_id)
        if record is None:
            return None
        barrier = self._ensure_hard_erasure_barrier(record)
        self._mirror_barrier(barrier)
        return barrier

    def _ensure_hard_erasure_barrier(
        self,
        record: DocumentEraseRecord,
    ) -> DocumentPublicationBarrier:
        current = self.control_store.load_publication_barrier(
            record.tenant_id,
            record.owner_user_id,
            record.document_id,
        )
        state = self._projection_state(
            record.tenant_id,
            record.owner_user_id,
            record.document_id,
        ) or {}
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
            _coerce_int(state.get("deletion_generation") or 0),
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
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> int:
        state = self._projection_state(tenant_id, owner_user_id, document_id) or {}
        floor = _coerce_int(state.get("projection_generation") or 0)
        for record in self._owner_document_records(tenant_id, owner_user_id):
            if record.document_id == document_id:
                floor = max(floor, int(record.projection_generation))
        return floor

    def _purge_document_derivatives(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        document_uri: str,
    ) -> None:
        """Replay exact identity cleanup after Catalog tombstoning or a crash."""

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

    def _projection_state(self, tenant: str, owner: str, document_id: str) -> Mapping[str, object] | None:
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

    def _mirror_barrier(self, barrier: DocumentPublicationBarrier) -> tuple[str, ...]:
        """Recreate or refresh the SQLite tombstone from protected control state."""

        state = self._projection_state(
            barrier.tenant_id,
            barrier.owner_user_id,
            barrier.document_id,
        )
        if state is not None:
            state_status = str(state.get("deletion_status") or "")
            state_deletion_generation = _coerce_int(state.get("deletion_generation") or 0)
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
            current_generation = _coerce_int(state.get("projection_generation") or 0)
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
            and _coerce_int(state.get("projection_generation") or 0)
            == control.projection_generation
            and str(state.get("source_digest") or "") == control.raw_sha256
            and str(state.get("relative_path") or "") == control.relative_path
            and not str(state.get("deletion_status") or "")
        )

    def _records(self, projection: MemoryDocumentProjection) -> tuple[CatalogRecord, tuple[CatalogRecord, ...]]:
        uri = MemoryDocumentPathPolicy.document_uri(projection.owner_user_id, projection.document_id)
        tree_path = self._tree_path(projection.relative_path)
        common: dict[str, Any] = {
            "tenant_id": projection.tenant_id,
            "owner_user_id": projection.owner_user_id,
            "context_type": ContextType.MEMORY.value,
            "source_kind": "markdown_memory_document",
            "lifecycle_state": "active",
            "primary_tree_path": tree_path,
            "tree_paths": (tree_path,),
            "source_uri": uri,
            "source_digest": projection.source_digest,
            "source_revision": projection.document_revision,
            "document_id": projection.document_id,
            "document_kind": projection.document_kind.value,
            "document_revision": projection.document_revision,
            "projection_generation": projection.projection_generation,
            "projection_effect_hash": projection.source_digest,
            "projection_status": CatalogProjectionStatus.PROJECTED.value,
            "metadata": {
                "relative_path": projection.relative_path,
                "source_authority": "live_markdown",
            },
        }
        document = CatalogRecord(
            record_key=f"memory-document:{projection.owner_user_id}:{projection.document_id}",
            uri=uri,
            record_kind=CatalogRecordKind.MEMORY_DOCUMENT.value,
            title=projection.title,
            l0_text=projection.l0_text,
            l1_text=projection.l1_text,
            l2_uri=uri,
            **common,
        )
        blocks = tuple(
            CatalogRecord(
                record_key=f"memory-block:{projection.owner_user_id}:{block.block_id}",
                uri=f"{uri}/blocks/{block.block_id}",
                record_kind=CatalogRecordKind.MEMORY_BLOCK.value,
                parent_uri=uri,
                title=" / ".join(block.heading_path) or projection.title,
                l0_text=" / ".join(block.heading_path),
                l1_text=block.text,
                l2_uri=uri,
                block_id=block.block_id,
                metadata={
                    **cast(dict[str, Any], common["metadata"]),
                    "heading_path": list(block.heading_path),
                    "occurrence": block.occurrence,
                },
                **{key: value for key, value in common.items() if key != "metadata"},
            )
            for block in projection.blocks
        )
        return document, blocks

    def _prepare_vector_rows(
        self,
        records: tuple[CatalogRecord, ...],
    ) -> tuple[tuple[str, list[float], dict[str, Any]], ...]:
        if self.vector_store is None or self.embedding_provider is None:
            return ()
        prepared: list[tuple[str, list[float], dict[str, Any]]] = []
        for record in records:
            text = "\n".join(part for part in (record.title, record.l0_text, record.l1_text) if part)
            if not text:
                continue
            embedding = [float(value) for value in self.embedding_provider.embed(text)]
            if not embedding or any(not math.isfinite(value) for value in embedding):
                raise ValueError("memory document embedding provider returned an invalid vector")
            metadata = {
                **catalog_vector_metadata(record),
                "record_key": record.record_key,
                "public_uri": record.uri,
                "embedding_model": str(getattr(self.embedding_provider, "model_name", "")),
                "schema_version": "memory_document_vector_v1",
            }
            prepared.append((vector_row_id(record.tenant_id, record.record_key), embedding, metadata))
        return tuple(prepared)

    def _replace_document_links(self, document_record: CatalogRecord, raw_bytes: bytes) -> None:
        if self.relation_store is None:
            return
        with self.erasure_store.owner_relation_lock(
            document_record.tenant_id,
            document_record.owner_user_id,
        ):
            self._replace_document_links_locked(document_record, raw_bytes)

    def _replace_document_links_locked(self, document_record: CatalogRecord, raw_bytes: bytes) -> None:
        assert self.relation_store is not None
        uri = document_record.uri
        while self.relation_store.delete_projection_relations(
            uri,
            tenant_id=document_record.tenant_id,
            catalog_record_key=document_record.record_key,
            limit=1_000,
        ):
            pass
        body = raw_bytes.decode("utf-8", errors="strict")
        targets_by_path = {
            str(record.metadata.get("relative_path") or ""): record
            for record in self._owner_document_records(
                document_record.tenant_id,
                document_record.owner_user_id,
            )
        }
        base = posixpath.dirname(str(document_record.metadata.get("relative_path") or ""))
        seen: set[str] = set()
        for match in _MARKDOWN_LINK.finditer(body):
            destination = match.group(1).strip().split("#", 1)[0]
            if not destination or "://" in destination or destination.startswith(("#", "/")):
                continue
            try:
                relative = MemoryDocumentPathPolicy.normalize_relative_path(
                    posixpath.normpath(posixpath.join(base, destination))
                )
            except ValueError:
                continue
            target_record = targets_by_path.get(relative)
            if target_record is None:
                continue
            target_uri = target_record.uri
            if target_uri == uri or target_uri in seen:
                continue
            target_document_id = target_record.document_id
            if self.erasure_store.load(
                document_record.tenant_id,
                document_record.owner_user_id,
                target_document_id,
            ) is not None:
                continue
            target_barrier = self.control_store.load_publication_barrier(
                document_record.tenant_id,
                document_record.owner_user_id,
                target_document_id,
            )
            target_state = self._projection_state(
                document_record.tenant_id,
                document_record.owner_user_id,
                target_document_id,
            ) or {}
            if str(target_state.get("deletion_status") or ""):
                continue
            if target_barrier is not None:
                target_control = self.control_store.load_control(
                    document_record.tenant_id,
                    document_record.owner_user_id,
                    target_document_id,
                )
                if not self._control_authorizes_restored_serving(
                    target_control,
                    target_barrier,
                    target_state,
                ):
                    continue
            seen.add(target_uri)
            self.relation_store.add_relation(
                ContextRelation(
                    source_uri=uri,
                    relation_type="links_to",
                    target_uri=target_uri,
                    metadata={
                        "tenant_id": document_record.tenant_id,
                        "owner_user_id": document_record.owner_user_id,
                        "catalog_record_key": document_record.record_key,
                        "projection_generation": document_record.projection_generation,
                        "source_digest": document_record.source_digest,
                    },
                ),
                tenant_id=document_record.tenant_id,
            )

    def _existing_projection_uris(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> dict[str, str]:
        list_catalog = getattr(self.catalog_store, "list_catalog", None)
        if not callable(list_catalog):
            return {}
        records = cast(_CatalogLister, list_catalog)(
            tenant_id=tenant_id,
            filters={"owner_user_id": owner_user_id, "document_id": document_id, "include_inactive": True},
            limit=1_000,
        )
        return {str(record.record_key): str(record.uri) for record in records}

    def _owner_document_records(self, tenant_id: str, owner_user_id: str) -> tuple[CatalogRecord, ...]:
        scanner = getattr(self.catalog_store, "scan_catalog_batch", None)
        if not callable(scanner):
            list_catalog = getattr(self.catalog_store, "list_catalog", None)
            if not callable(list_catalog):
                raise RuntimeError("Catalog store has no bounded projection scanner")
            listed_records = cast(_CatalogLister, list_catalog)(
                tenant_id=tenant_id,
                filters={
                    "owner_user_id": owner_user_id,
                    "record_kind": CatalogRecordKind.MEMORY_DOCUMENT.value,
                    "include_inactive": True,
                },
                limit=1_000,
            )
            return tuple(listed_records)
        scan_batch = cast(_CatalogBatchScanner, scanner)
        records: list[CatalogRecord] = []
        cursor = ""
        while True:
            batch = scan_batch(
                tenant_id=tenant_id,
                after_record_key=cursor,
                filters={
                    "owner_user_id": owner_user_id,
                    "record_kind": CatalogRecordKind.MEMORY_DOCUMENT.value,
                    "include_inactive": True,
                },
                limit=256,
            )
            if not batch:
                return tuple(records)
            records.extend(batch)
            next_cursor = str(batch[-1].record_key)
            if next_cursor <= cursor:
                raise RuntimeError("Catalog document scan did not advance")
            cursor = next_cursor

    def _remove_obsolete(
        self,
        tenant_id: str,
        record_keys: tuple[str, ...],
        existing_uris: Mapping[str, str],
    ) -> None:
        for record_key in record_keys:
            if self.vector_store is not None:
                self.vector_store.delete_vector(vector_row_id(tenant_id, record_key))
                # The row ID is deterministic for conforming writers, while
                # the exact metadata identity also removes legacy/external
                # rows whose physical key did not follow that convention.
                deleted = self.vector_store.delete_by_filter(
                    {
                        "tenant_id": tenant_id,
                        "catalog_record_key": record_key,
                    }
                )
                if not isinstance(deleted, int) or isinstance(deleted, bool) or deleted < 0:
                    raise TypeError("vector delete-by-filter returned an invalid deletion count")
            # Relations created from a projection carry the exact Catalog key.
            if self.relation_store is not None:
                uri = str(existing_uris.get(record_key) or "")
                if not uri:
                    continue
                while self.relation_store.delete_projection_relations(
                    uri,
                    tenant_id=tenant_id,
                    catalog_record_key=record_key,
                    limit=1_000,
                ):
                    pass

    @staticmethod
    def _tree_path(relative_path: str) -> str:
        relative = MemoryDocumentPathPolicy.normalize_relative_path(relative_path)
        if relative == "MEMORY.md":
            return "memories/root"
        if relative == "profile.md":
            return "memories/profile"
        if relative == "preferences.md":
            return "memories/preferences"
        if relative == "knowledge/MEMORY.md":
            return "memories/knowledge"
        if relative == "knowledge/open-loops.md":
            return "memories/knowledge/open-loops"
        stem = relative.removesuffix(".md")
        return f"memories/{stem}"

    @staticmethod
    def _deletion_digest(payload: Mapping[str, object]) -> str:
        return deletion_event_digest(
            event_id=str(payload["event_id"]),
            document_id=str(payload["document_id"]),
            before_raw_digest=str(payload["before_raw_digest"]),
            projection_generation=_coerce_int(payload["projection_generation"]),
        )

    @staticmethod
    def _payload(job: QueueJob) -> Mapping[str, object]:
        if job.queue_name != "memory_projection" or job.action != "memory_committed":
            raise ValueError("job is not a Markdown memory projection event")
        payload = dict(job.payload)
        required = {
            "schema",
            "tenant_id",
            "owner_user_id",
            "document_id",
            "intent_id",
            "event_id",
            "edit_kind",
            "old_relative_path",
            "new_relative_path",
            "before_raw_digest",
            "after_raw_digest",
            "logical_revision",
            "projection_generation",
        }
        if set(payload) != required or payload.get("schema") != "memory_document_projection_v1":
            raise ValueError("memory projection queue payload schema is invalid")
        tenant = MemoryDocumentPathPolicy.trusted_segment(payload["tenant_id"], "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(payload["owner_user_id"], "owner_user_id")
        document_id = str(payload["document_id"])
        expected_uri = MemoryDocumentPathPolicy.document_uri(owner, document_id)
        if job.target_uri != expected_uri:
            raise ValueError("memory projection queue target is detached from its document")
        DocumentEditKind(str(payload["edit_kind"]))
        if int(payload["logical_revision"]) <= 0 or int(payload["projection_generation"]) <= 0:
            raise ValueError("memory projection queue generation is invalid")
        payload["tenant_id"] = tenant
        payload["owner_user_id"] = owner
        return payload


class MemoryDocumentCatalogEraseBackend:
    """Hard-erase Catalog/FTS/path and attached Vector/Relation projections."""

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
        current = self.worker._projection_state(
            request.tenant_id,
            request.owner_user_id,
            request.document_id,
        ) or {}
        if not (
            current.get("deletion_status") == "HARD_ERASED"
            and current.get("deletion_event_digest") == barrier.deletion_event_digest
            and _coerce_int(current.get("deletion_generation") or 0) == barrier.deletion_generation
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
            and _coerce_int(state.get("deletion_generation") or 0) == barrier.deletion_generation
            and not state.get("source_digest")
        )


__all__ = [
    "MemoryDocumentCatalogEraseBackend",
    "MemoryDocumentProjectionWorker",
    "MemoryProjectionRun",
]
