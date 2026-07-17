"""Lifecycle responsibilities for canonical projection."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from memoryos.contextdb.store.queue_store import (
    QueueJob,
    QueueLeaseIdentityError,
)
from memoryos.core.readiness import (
    readiness_for_source_store,
    require_source_store_ready,
    require_source_store_recovering,
)
from memoryos.memory.canonical.projection_proof import (
    AuthoritativeProjectionIntegrityError,
)

from .models import (
    ProjectionOutboxIntegrityError,
)

if TYPE_CHECKING:
    from .worker import MemoryProjectionWorker


def process_pending(
    self: MemoryProjectionWorker,
    limit: int = 10,
    *,
    lease_seconds: int = 60,
    max_retries: int = 3,
) -> dict[str, list[str]]:
    with self._migration_projection_fence():
        require_source_store_ready(self.projector.source_store)
        return self._process_pending(
            limit,
            lease_seconds=lease_seconds,
            max_retries=max_retries,
        )


def _process_pending_during_startup(
    self: MemoryProjectionWorker,
    limit: int = 10,
    *,
    lease_seconds: int = 60,
    max_retries: int = 3,
) -> dict[str, list[str]]:
    with self._migration_projection_fence():
        require_source_store_recovering(self.projector.source_store)
        return self._process_pending(
            limit,
            lease_seconds=lease_seconds,
            max_retries=max_retries,
        )


def _process_pending(
    self: MemoryProjectionWorker,
    limit: int,
    *,
    lease_seconds: int,
    max_retries: int,
) -> dict[str, list[str]]:
    self.last_quarantined = []
    self._validate_authoritative_projection_proofs()
    self.dispatch_outbox()
    processed: list[str] = []
    stale: list[str] = []
    failed: list[str] = []
    dead_letter: list[str] = []
    quarantine: list[str] = []
    released: list[str] = []
    jobs = self.queue_store.lease(
        "memory_projection",
        lease_owner=self.worker_id,
        limit=limit,
        lease_seconds=lease_seconds,
    )
    for position, job in enumerate(jobs):
        try:
            outbox = self._load_projection_job_outbox(job)
            self._project_event(outbox, job.job_id, stale)
            self._assert_projection_job_identity_unchanged(job)
            self._ensure_projection_publication(outbox, job)
            self._assert_projection_job_identity_unchanged(job)
            self.queue_store.ack(job)
        except QueueLeaseIdentityError as exc:
            self._mark_authoritative_integrity_failure(
                exc,
                artifact="projection_queue",
                identifiers={"job_id": job.job_id},
            )
            released.extend(
                self._release_unattempted_projection_jobs(
                    jobs[position + 1 :],
                    cause=type(exc).__name__,
                )
            )
            self._quarantine_projection_identity_conflict(job, exc)
            failed.append(job.job_id)
            quarantine.append(job.job_id)
            break
        except (ProjectionOutboxIntegrityError, AuthoritativeProjectionIntegrityError) as exc:
            self._mark_authoritative_integrity_failure(
                exc,
                artifact=(
                    "projection_proof"
                    if isinstance(exc, AuthoritativeProjectionIntegrityError)
                    else "projection_outbox_or_queue"
                ),
                identifiers={"job_id": job.job_id},
            )
            released.extend(
                self._release_unattempted_projection_jobs(
                    jobs[position + 1 :],
                    cause=type(exc).__name__,
                )
            )
            self.queue_store.quarantine(job, type(exc).__name__)
            failed.append(job.job_id)
            quarantine.append(job.job_id)
            break
        except Exception as exc:
            settled = self.queue_store.retry(
                job,
                type(exc).__name__,
                max_retries=max_retries,
                retryable=True,
            )
            failed.append(job.job_id)
            if settled.status == "dead_letter":
                dead_letter.append(job.job_id)
                self._mark_authoritative_integrity_failure(
                    exc,
                    artifact="projection_queue_dead_letter",
                    identifiers={"job_id": job.job_id},
                )
                released.extend(
                    self._release_unattempted_projection_jobs(
                        jobs[position + 1 :],
                        cause="projection_queue_dead_letter",
                    )
                )
                break
            self._extend_unattempted_projection_leases(
                jobs[position + 1 :],
                lease_seconds=lease_seconds,
            )
            continue
        processed.append(job.job_id)
        self._extend_unattempted_projection_leases(
            jobs[position + 1 :],
            lease_seconds=lease_seconds,
        )
    return {
        "processed": processed,
        "stale": stale,
        "failed": failed,
        "dead_letter": dead_letter,
        "quarantine": [*self.last_quarantined, *quarantine],
        "released": released,
    }


def _validate_authoritative_projection_proofs(self: MemoryProjectionWorker) -> None:
    """Reverse-bind immutable proofs without inspecting rebuildable views."""

    try:
        self.proof_store.validate_all()
        for publication in self.proof_store.iter_publications():
            transaction_id = str(publication["transaction_id"])
            job = self.queue_store.get(f"outbox_{transaction_id}")
            if job is None:
                raise AuthoritativeProjectionIntegrityError(
                    "projection publication receipt has no durable queue identity"
                )
            outbox = self._load_projection_job_outbox(
                job,
                expected_transaction_id=transaction_id,
            )
            receipt = self._load_bound_receipt(
                outbox,
                transaction_id,
                str(publication["commit_group_id"]),
            )
            self._verify_projection_publication_boundary(
                publication,
                outbox,
                receipt,
                job,
            )
            completion = self.proof_store.load_completion(transaction_id)
            if completion is not None and job.status != "done":
                raise AuthoritativeProjectionIntegrityError(
                    "projection completion proof is detached from terminal queue state"
                )
    except (AuthoritativeProjectionIntegrityError, ProjectionOutboxIntegrityError) as exc:
        self._mark_authoritative_integrity_failure(
            exc,
            artifact=(
                "projection_proof"
                if isinstance(exc, AuthoritativeProjectionIntegrityError)
                else "projection_outbox_or_queue"
            ),
        )
        raise


def _mark_authoritative_integrity_failure(
    self: MemoryProjectionWorker,
    error: BaseException,
    *,
    artifact: str,
    identifiers: dict[str, Any] | None = None,
) -> None:
    readiness = readiness_for_source_store(self.projector.source_store)
    mark_not_ready = getattr(readiness, "mark_not_ready", None)
    if not callable(mark_not_ready):
        return
    details: dict[str, Any] = {
        "artifact": artifact,
        "error_type": type(error).__name__,
        **dict(identifiers or {}),
    }
    mark_not_ready(
        f"authoritative projection integrity failure: {type(error).__name__}: {error}",
        details=details,
    )


def _release_unattempted_projection_jobs(
    self: MemoryProjectionWorker,
    jobs: list[QueueJob],
    *,
    cause: str,
) -> list[str]:
    """Release the remainder of an already-leased batch without retry cost."""

    released: list[str] = []
    for job in jobs:
        try:
            settled = self.queue_store.release(
                job,
                f"batch aborted before attempt after {cause}",
            )
        except Exception as exc:
            self._mark_authoritative_integrity_failure(
                exc,
                artifact="projection_queue_release",
                identifiers={"job_id": job.job_id},
            )
            raise ProjectionOutboxIntegrityError(
                "projection batch abort could not release an unattempted lease"
            ) from exc
        if settled.status != "pending":
            error = ProjectionOutboxIntegrityError("projection batch abort released a job to a non-pending state")
            self._mark_authoritative_integrity_failure(
                error,
                artifact="projection_queue_release",
                identifiers={"job_id": job.job_id},
            )
            raise error
        released.append(job.job_id)
    return released


def _extend_unattempted_projection_leases(
    self: MemoryProjectionWorker,
    jobs: list[QueueJob],
    *,
    lease_seconds: int,
) -> None:
    """Keep an already leased fail-stop batch owned while earlier work runs."""

    for job in jobs:
        self.queue_store.extend(job, lease_seconds=lease_seconds)


def _assert_projection_job_identity_unchanged(self: MemoryProjectionWorker, job: QueueJob) -> None:
    """Re-read a leased job so post-preflight queue tamper cannot publish."""

    try:
        persisted = self.queue_store.get(job.job_id)
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise QueueLeaseIdentityError(f"projection queue identity is unreadable while leased: {job.job_id}") from exc
    if persisted is None or (
        persisted.queue_name != job.queue_name
        or persisted.action != job.action
        or persisted.target_uri != job.target_uri
        or persisted.payload != job.payload
    ):
        raise QueueLeaseIdentityError(f"projection queue immutable identity changed while leased: {job.job_id}")


def _quarantine_projection_identity_conflict(
    self: MemoryProjectionWorker,
    job: QueueJob,
    error: QueueLeaseIdentityError,
) -> None:
    try:
        settled = self.queue_store.quarantine_identity_conflict(
            job,
            type(error).__name__,
        )
    except Exception as exc:
        self._mark_authoritative_integrity_failure(
            exc,
            artifact="projection_queue_quarantine",
            identifiers={"job_id": job.job_id},
        )
        raise ProjectionOutboxIntegrityError("corrupt projection queue identity could not be quarantined") from exc
    if settled.status != "quarantine":
        failure = ProjectionOutboxIntegrityError("corrupt projection queue identity was not quarantined")
        self._mark_authoritative_integrity_failure(
            failure,
            artifact="projection_queue_quarantine",
            identifiers={"job_id": job.job_id},
        )
        raise failure


def process_commit_group(
    self: MemoryProjectionWorker,
    group_id: str,
    *,
    transaction_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    with self._migration_projection_fence():
        return self._process_commit_group_unfenced(
            group_id,
            transaction_ids=transaction_ids,
        )


def _process_commit_group_unfenced(
    self: MemoryProjectionWorker,
    group_id: str,
    *,
    transaction_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Project only one durable commit group, independently of unrelated queue jobs."""

    readiness = readiness_for_source_store(self.projector.source_store)
    state_obj = getattr(readiness, "state", None)
    state = str(getattr(state_obj, "value", state_obj or ""))
    if state != "RECOVERING":
        require_source_store_ready(self.projector.source_store)
    self._validate_authoritative_projection_proofs()

    processed: list[str] = []
    stale: list[str] = []
    failed: list[str] = []
    quarantine: list[str] = []
    released: list[str] = []
    completion_proofs: list[dict[str, Any]] = []
    terminal_abort = False
    self.last_quarantined = []
    self.dispatch_outbox()
    if transaction_ids:
        job_ids = tuple(f"outbox_{transaction_id}" for transaction_id in transaction_ids)
    else:
        outbox_root = self.projector.root / "system" / "outbox"
        selected: list[str] = []
        for path in sorted(outbox_root.glob("*.json")) if outbox_root.exists() else []:
            try:
                event = self._read_outbox(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if str(event.get("commit_group_id", "")) == group_id:
                selected.append(f"outbox_{path.stem}")
        job_ids = tuple(selected)
    if not job_ids:
        return {
            "processed": processed,
            "stale": stale,
            "failed": failed,
            "quarantine": self.last_quarantined,
            "released": released,
        }
    lease_seconds = 300
    jobs = self.queue_store.lease(
        "memory_projection",
        lease_owner=self.worker_id,
        limit=len(job_ids),
        lease_seconds=lease_seconds,
        job_ids=job_ids,
    )
    for position, job in enumerate(jobs):
        try:
            outbox = self._load_projection_job_outbox(job)
            if str(outbox.get("commit_group_id", "")) != group_id:
                if transaction_ids:
                    raise ValueError("projection outbox is not bound to the requested commit group")
                released.extend(
                    self._release_unattempted_projection_jobs(
                        [job],
                        cause="commit_group_filter_mismatch",
                    )
                )
                continue
            self._project_event(outbox, job.job_id, stale)
            self._assert_projection_job_identity_unchanged(job)
            self._ensure_projection_publication(outbox, job)
            self._assert_projection_job_identity_unchanged(job)
            self.queue_store.ack(job)
            processed.append(job.job_id)
        except QueueLeaseIdentityError as exc:
            self._mark_authoritative_integrity_failure(
                exc,
                artifact="projection_queue",
                identifiers={"job_id": job.job_id, "commit_group_id": group_id},
            )
            released.extend(
                self._release_unattempted_projection_jobs(
                    jobs[position + 1 :],
                    cause=type(exc).__name__,
                )
            )
            self._quarantine_projection_identity_conflict(job, exc)
            failed.append(f"{job.job_id}:{type(exc).__name__}")
            quarantine.append(job.job_id)
            terminal_abort = True
            break
        except (ProjectionOutboxIntegrityError, AuthoritativeProjectionIntegrityError) as exc:
            self._mark_authoritative_integrity_failure(
                exc,
                artifact=(
                    "projection_proof"
                    if isinstance(exc, AuthoritativeProjectionIntegrityError)
                    else "projection_outbox_or_queue"
                ),
                identifiers={"job_id": job.job_id, "commit_group_id": group_id},
            )
            released.extend(
                self._release_unattempted_projection_jobs(
                    jobs[position + 1 :],
                    cause=type(exc).__name__,
                )
            )
            self.queue_store.quarantine(job, type(exc).__name__)
            failed.append(f"{job.job_id}:{type(exc).__name__}")
            quarantine.append(job.job_id)
            terminal_abort = True
            break
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            settled = self.queue_store.retry(job, type(exc).__name__, max_retries=3, retryable=True)
            if settled.status == "dead_letter":
                self._mark_authoritative_integrity_failure(
                    exc,
                    artifact="projection_queue_dead_letter",
                    identifiers={"job_id": job.job_id, "commit_group_id": group_id},
                )
                released.extend(
                    self._release_unattempted_projection_jobs(
                        jobs[position + 1 :],
                        cause="projection_queue_dead_letter",
                    )
                )
                terminal_abort = True
            failed.append(f"{job.job_id}:{type(exc).__name__}")
            if terminal_abort:
                failed.append(f"{job.job_id}:queue_dead_letter")
                break
            self._extend_unattempted_projection_leases(
                jobs[position + 1 :],
                lease_seconds=lease_seconds,
            )
        except Exception as exc:
            settled = self.queue_store.retry(job, type(exc).__name__, max_retries=3, retryable=False)
            if settled.status == "dead_letter":
                self._mark_authoritative_integrity_failure(
                    exc,
                    artifact="projection_queue_dead_letter",
                    identifiers={"job_id": job.job_id, "commit_group_id": group_id},
                )
                released.extend(
                    self._release_unattempted_projection_jobs(
                        jobs[position + 1 :],
                        cause="projection_queue_dead_letter",
                    )
                )
                terminal_abort = True
            failed.append(f"{job.job_id}:{type(exc).__name__}")
            if terminal_abort:
                failed.append(f"{job.job_id}:queue_dead_letter")
                break
            self._extend_unattempted_projection_leases(
                jobs[position + 1 :],
                lease_seconds=lease_seconds,
            )
        else:
            self._extend_unattempted_projection_leases(
                jobs[position + 1 :],
                lease_seconds=lease_seconds,
            )
    if transaction_ids and not terminal_abort:
        completion = self.verify_commit_group_completion(group_id, transaction_ids)
        failed.extend(completion["failures"])
        completion_proofs.extend(completion["proofs"])
    return {
        "processed": processed,
        "stale": stale,
        "failed": failed,
        "quarantine": [*self.last_quarantined, *quarantine],
        "completion_proofs": completion_proofs,
        "released": released,
    }
