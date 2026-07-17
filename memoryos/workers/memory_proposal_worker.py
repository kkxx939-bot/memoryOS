"""后台任务里的记忆提案任务。"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from memoryos.application.session.commit_service import SessionCommitService
from memoryos.contextdb.session.session_model import SessionCommitState
from memoryos.contextdb.store.queue_store import QueueJob
from memoryos.core.readiness import require_session_service_ready, session_service_is_ready
from memoryos.memory.canonical.salience_ledger import SalienceLedgerIntegrityError
from memoryos.memory.integration.planning_envelope import PlanningEnvelopeIntegrityError
from memoryos.workers.tenant_boundary import require_bound_job_tenant


class MemoryProposalWorker:
    """跑 MemoryProposalWorker 对应的后台任务。"""

    def __init__(self, service: SessionCommitService, *, worker_id: str | None = None) -> None:
        self.service = service
        self.worker_id = worker_id or f"memory-proposal:{os.getpid()}:{uuid.uuid4().hex}"

    @contextmanager
    def _migration_projection_fence(self) -> Iterator[None]:
        gate: Any | None = getattr(self.service, "migration_gate", None)
        acquire = getattr(gate, "acquire_projection_fence", None)
        release = getattr(gate, "release_projection_fence", None)
        fence = acquire() if callable(acquire) else None
        try:
            yield
        finally:
            if callable(release):
                release(fence)

    def process_pending(self, *, batch_size: int = 10, lease_seconds: int = 60, max_retries: int = 3) -> dict:
        with self._migration_projection_fence():
            return self._process_pending_unfenced(
                batch_size=batch_size,
                lease_seconds=lease_seconds,
                max_retries=max_retries,
            )

    def _process_pending_unfenced(
        self,
        *,
        batch_size: int,
        lease_seconds: int,
        max_retries: int,
    ) -> dict:
        require_session_service_ready(self.service)
        committed = failed = dead_letter = 0
        released: list[str] = []
        jobs = self.service.queue_store.lease(
            "memory_proposal",
            lease_owner=self.worker_id,
            limit=batch_size,
            lease_seconds=lease_seconds,
        )
        for position, job in enumerate(jobs):
            try:
                require_session_service_ready(self.service)
            except RuntimeError:
                if session_service_is_ready(self.service):
                    raise
                released.extend(self._release_unattempted(jobs[position:]))
                break
            try:
                tenant_id = require_bound_job_tenant(
                    job.payload,
                    bound_tenant_id=self.service.archive_store.tenant_id,
                )
                archive = self.service.archive_store.read_archive(
                    job.target_uri,
                    tenant_id=tenant_id,
                    manifest_digest=str(job.payload.get("manifest_digest") or "") or None,
                )
                result = self.service.async_commit(archive)
                if result.canonical_committed:
                    if not session_service_is_ready(self.service):
                        failed += 1
                        released.extend(self._release_unattempted(jobs[position:]))
                        break
                    self.service.queue_store.ack(job)
                    committed += 1
                    continue
                if not session_service_is_ready(self.service):
                    failed += 1
                    released.extend(self._release_unattempted(jobs[position:]))
                    break
                retryable = result.state != SessionCommitState.DEAD_LETTER
                status = self.service.queue_store.retry(
                    job,
                    result.state.value,
                    max_retries=max_retries,
                    retryable=retryable,
                ).status
                failed += 1
                dead_letter += int(status == "dead_letter")
            except (PlanningEnvelopeIntegrityError, SalienceLedgerIntegrityError) as exc:
                if not session_service_is_ready(self.service):
                    failed += 1
                    released.extend(self._release_unattempted(jobs[position:]))
                    break
                status = self.service.queue_store.retry(
                    job,
                    type(exc).__name__,
                    max_retries=max_retries,
                    retryable=False,
                ).status
                failed += 1
                dead_letter += int(status == "dead_letter")
            except (OSError, TimeoutError, RuntimeError) as exc:
                if not session_service_is_ready(self.service):
                    failed += 1
                    released.extend(self._release_unattempted(jobs[position:]))
                    break
                status = self.service.queue_store.retry(
                    job,
                    type(exc).__name__,
                    max_retries=max_retries,
                    retryable=True,
                ).status
                failed += 1
                dead_letter += int(status == "dead_letter")
            except (ValueError, KeyError, TypeError) as exc:
                if not session_service_is_ready(self.service):
                    failed += 1
                    released.extend(self._release_unattempted(jobs[position:]))
                    break
                status = self.service.queue_store.retry(
                    job,
                    type(exc).__name__,
                    max_retries=max_retries,
                    retryable=False,
                ).status
                failed += 1
                dead_letter += int(status == "dead_letter")
            except Exception as exc:
                # SessionCommitService terminalizes its canonical lease before
                # propagating an unclassified failure.  The worker must also
                # surrender its independent queue lease and must not invent a
                # retry policy for an unknown error class.
                if not session_service_is_ready(self.service):
                    failed += 1
                    released.extend(self._release_unattempted(jobs[position:]))
                    break
                status = self.service.queue_store.retry(
                    job,
                    type(exc).__name__,
                    max_retries=max_retries,
                    retryable=False,
                ).status
                failed += 1
                dead_letter += int(status == "dead_letter")
        summary: dict[str, object] = {
            "claimed": len(jobs),
            "committed": committed,
            "failed": failed,
            "dead_letter": dead_letter,
        }
        if released:
            summary["released"] = released
        if not session_service_is_ready(self.service):
            summary["status"] = "not_ready"
        return summary

    def _release_unattempted(self, jobs: list[QueueJob]) -> list[str]:
        """Fenced release for jobs not allowed to proceed after NOT_READY."""

        released: list[str] = []
        for job in jobs:
            settled = self.service.queue_store.release(job)
            if (
                settled.status != "pending"
                or settled.retry_count != job.retry_count
                or settled.lease_token
                or settled.lease_owner
            ):
                raise RuntimeError("memory proposal batch release did not preserve queue state")
            released.append(job.job_id)
        return released
