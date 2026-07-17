"""Stable worker facade for canonical projection."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from memoryos.contextdb.store.queue_store import (
    QueueJob,
    QueueLeaseIdentityError,
    QueueStore,
)
from memoryos.memory.canonical.projection_proof import (
    ProjectionProofStore,
)
from memoryos.memory.canonical.projection_state import (
    ProjectionRecord,
)
from memoryos.memory.canonical.slot_projection import CurrentSlotProjection, CurrentSlotProjectionResult

from . import historical as _historical
from . import lifecycle as _lifecycle
from . import outbox as _outbox
from . import proofs as _proofs
from .models import _CurrentSlotProjectionTarget
from .service import CanonicalMemoryProjector


class MemoryProjectionWorker:
    """Consume durable MemoryCommitted outbox entries idempotently."""

    def __init__(
        self,
        projector: CanonicalMemoryProjector,
        queue_store: QueueStore,
        *,
        current_slot_projector: CurrentSlotProjection | None = None,
        migration_gate: Any = None,
        worker_id: str | None = None,
    ) -> None:
        self.projector = projector
        self.queue_store = queue_store
        self.current_slot_projector = current_slot_projector
        self.migration_gate = migration_gate
        self.proof_store = ProjectionProofStore(projector.root)
        self.worker_id = worker_id or f"memory-projection:{os.getpid()}:{uuid.uuid4().hex}"
        self.last_quarantined: list[str] = []
        self._projection_fence_depth: ContextVar[int] = ContextVar(
            f"memoryos_projection_worker_fence_depth_{id(self)}",
            default=0,
        )

    @contextmanager
    def _migration_projection_fence(self) -> Iterator[None]:
        """Hold the tenant rebuild fence before dispatching or leasing work.

        ``process_commit_group`` can be reached from startup/session recovery
        while another guarded projection entry is already active.  Keep the
        guard execution-context reentrant so a non-reentrant SQLite lease is
        acquired exactly once.
        """

        depth = self._projection_fence_depth.get()
        if depth:
            depth_token = self._projection_fence_depth.set(depth + 1)
            try:
                yield
            finally:
                self._projection_fence_depth.reset(depth_token)
            return
        acquire = getattr(self.migration_gate, "acquire_projection_fence", None)
        release = getattr(self.migration_gate, "release_projection_fence", None)
        fence = acquire() if callable(acquire) else None
        depth_token = self._projection_fence_depth.set(1)
        try:
            yield
        finally:
            self._projection_fence_depth.reset(depth_token)
            if callable(release):
                release(fence)

    def process_pending(
        self,
        limit: int = 10,
        *,
        lease_seconds: int = 60,
        max_retries: int = 3,
    ) -> dict[str, list[str]]:
        return _lifecycle.process_pending(self, limit, lease_seconds=lease_seconds, max_retries=max_retries)

    def _process_pending_during_startup(
        self,
        limit: int = 10,
        *,
        lease_seconds: int = 60,
        max_retries: int = 3,
    ) -> dict[str, list[str]]:
        return _lifecycle._process_pending_during_startup(
            self, limit, lease_seconds=lease_seconds, max_retries=max_retries
        )

    def _process_pending(
        self,
        limit: int,
        *,
        lease_seconds: int,
        max_retries: int,
    ) -> dict[str, list[str]]:
        return _lifecycle._process_pending(self, limit, lease_seconds=lease_seconds, max_retries=max_retries)

    def _validate_authoritative_projection_proofs(self) -> None:
        return _lifecycle._validate_authoritative_projection_proofs(self)

    def _mark_authoritative_integrity_failure(
        self,
        error: BaseException,
        *,
        artifact: str,
        identifiers: dict[str, Any] | None = None,
    ) -> None:
        return _lifecycle._mark_authoritative_integrity_failure(self, error, artifact=artifact, identifiers=identifiers)

    def _release_unattempted_projection_jobs(
        self,
        jobs: list[QueueJob],
        *,
        cause: str,
    ) -> list[str]:
        return _lifecycle._release_unattempted_projection_jobs(self, jobs, cause=cause)

    def _extend_unattempted_projection_leases(
        self,
        jobs: list[QueueJob],
        *,
        lease_seconds: int,
    ) -> None:
        return _lifecycle._extend_unattempted_projection_leases(self, jobs, lease_seconds=lease_seconds)

    def _assert_projection_job_identity_unchanged(self, job: QueueJob) -> None:
        return _lifecycle._assert_projection_job_identity_unchanged(self, job)

    def _quarantine_projection_identity_conflict(
        self,
        job: QueueJob,
        error: QueueLeaseIdentityError,
    ) -> None:
        return _lifecycle._quarantine_projection_identity_conflict(self, job, error)

    def verify_current_projections(self) -> dict[str, int]:
        return _proofs.verify_current_projections(self)

    def validate_projection_proofs(self) -> dict[str, int]:
        return _proofs.validate_projection_proofs(self)

    def migrate_legacy_completion_proof(
        self,
        group_id: str,
        transaction_id: str,
        legacy_proof: dict[str, Any],
    ) -> bool:
        return _proofs.migrate_legacy_completion_proof(self, group_id, transaction_id, legacy_proof)

    def _migrated_legacy_claim_proof(
        self,
        legacy: dict[str, Any],
        receipt: dict[str, Any],
        legacy_digest: str,
    ) -> dict[str, Any]:
        return _proofs._migrated_legacy_claim_proof(self, legacy, receipt, legacy_digest)

    def process_commit_group(
        self,
        group_id: str,
        *,
        transaction_ids: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        return _lifecycle.process_commit_group(self, group_id, transaction_ids=transaction_ids)

    def _process_commit_group_unfenced(
        self,
        group_id: str,
        *,
        transaction_ids: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        return _lifecycle._process_commit_group_unfenced(self, group_id, transaction_ids=transaction_ids)

    def _verify_projection_completion(
        self,
        group_id: str,
        transaction_ids: tuple[str, ...],
    ) -> list[str]:
        return _proofs._verify_projection_completion(self, group_id, transaction_ids)

    def verify_commit_group_completion(
        self,
        group_id: str,
        transaction_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        return _proofs.verify_commit_group_completion(self, group_id, transaction_ids)

    def _ensure_projection_publication(
        self,
        outbox: dict[str, Any],
        job: QueueJob,
    ) -> dict[str, Any]:
        return _proofs._ensure_projection_publication(self, outbox, job)

    def _verify_projection_publication(
        self,
        publication: dict[str, Any],
        outbox: dict[str, Any],
        receipt: dict[str, Any],
        job: QueueJob,
    ) -> None:
        return _proofs._verify_projection_publication(self, publication, outbox, receipt, job)

    def _verify_projection_publication_boundary(
        self,
        publication: dict[str, Any],
        outbox: dict[str, Any],
        receipt: dict[str, Any],
        job: QueueJob,
    ) -> None:
        return _proofs._verify_projection_publication_boundary(self, publication, outbox, receipt, job)

    def _load_bound_receipt(
        self,
        outbox: dict[str, Any],
        transaction_id: str,
        group_id: str,
    ) -> dict[str, Any]:
        return _proofs._load_bound_receipt(self, outbox, transaction_id, group_id)

    @staticmethod
    def _queue_identity_digest(job: QueueJob) -> str:
        return _proofs._queue_identity_digest(job)

    @staticmethod
    def _claim_revisions(outbox: dict[str, Any]) -> list[dict[str, Any]]:
        return _proofs._claim_revisions(outbox)

    def _verify_claim_projection(self, claim_uri: str, source_revision: int) -> dict[str, Any]:
        return _proofs._verify_claim_projection(self, claim_uri, source_revision)

    def _verify_historical_claim_projection(
        self,
        claim_proof: dict[str, Any],
        receipt: dict[str, Any],
    ) -> None:
        return _historical._verify_historical_claim_projection(self, claim_proof, receipt)

    def _materialize_historical_claim_projection(
        self,
        receipt: dict[str, Any],
        claim_uri: str,
        source_revision: int,
    ) -> dict[str, Any]:
        return _historical._materialize_historical_claim_projection(self, receipt, claim_uri, source_revision)

    @staticmethod
    def _historical_component_attestation(
        component: str,
        *,
        claim_uri: str,
        source_revision: int,
        transaction_id: str,
        receipt_digest: str,
    ) -> str:
        return _historical._historical_component_attestation(
            component,
            claim_uri=claim_uri,
            source_revision=source_revision,
            transaction_id=transaction_id,
            receipt_digest=receipt_digest,
        )

    def _restore_historical_projection_record(self, record: ProjectionRecord) -> None:
        return _historical._restore_historical_projection_record(self, record)

    def _matching_current_views(
        self,
        kind: str,
        record: ProjectionRecord,
        domain_identity: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return _historical._matching_current_views(self, kind, record, domain_identity)

    @staticmethod
    def _assert_projection_identity(
        payload: dict[str, Any],
        record: ProjectionRecord,
        *,
        label: str,
        domain_identity: dict[str, Any],
    ) -> None:
        return _historical._assert_projection_identity(payload, record, label=label, domain_identity=domain_identity)

    def _read_outbox(self, path: Path) -> dict[str, Any]:
        return _outbox._read_outbox(self, path)

    def _load_projection_job_outbox(
        self,
        job: QueueJob,
        *,
        expected_transaction_id: str = "",
    ) -> dict[str, Any]:
        return _outbox._load_projection_job_outbox(self, job, expected_transaction_id=expected_transaction_id)

    def _project_event(self, outbox: dict[str, Any], job_id: str, stale: list[str]) -> None:
        return _outbox._project_event(self, outbox, job_id, stale)

    def _record_current_slot_equivalence(
        self,
        outbox: dict[str, Any],
        target: _CurrentSlotProjectionTarget,
        result: CurrentSlotProjectionResult,
    ) -> None:
        return _outbox._record_current_slot_equivalence(self, outbox, target, result)

    @staticmethod
    def _current_slot_projection_targets(
        outbox: dict[str, Any],
    ) -> tuple[_CurrentSlotProjectionTarget, ...]:
        return _outbox._current_slot_projection_targets(outbox)

    @staticmethod
    def _optional_claim_id(value: object, *, label: str) -> str | None:
        return _outbox._optional_claim_id(value, label=label)

    def dispatch_outbox(self) -> list[str]:
        return _outbox.dispatch_outbox(self)

    def _dispatch_outbox_unfenced(self) -> list[str]:
        return _outbox._dispatch_outbox_unfenced(self)
