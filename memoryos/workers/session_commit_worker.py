"""后台任务里的会话提交任务。"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from memoryos.application.session.commit_service import SessionCommitService
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.queue_store import QueueJob
from memoryos.core.readiness import require_session_service_ready, session_service_is_ready
from memoryos.memory.canonical.salience_ledger import SalienceLedgerIntegrityError
from memoryos.memory.integration.planning_envelope import PlanningEnvelopeIntegrityError
from memoryos.operations.commit.commit_group import CommitGroupIntegrityError
from memoryos.workers.tenant_boundary import require_bound_job_tenant


class SessionCommitWorker:
    def __init__(self, service: SessionCommitService, *, worker_id: str | None = None) -> None:
        self.service = service
        self.worker_id = worker_id or f"session-commit:{os.getpid()}:{uuid.uuid4().hex}"

    def process_archive(self, archive: SessionArchive) -> dict:
        require_session_service_ready(self.service)
        result = self.service.async_commit(archive)
        return {"task_id": result.task_id, "status": result.status, "done": result.done}

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
        # The readiness check precedes commit-group recovery and queue leasing:
        # a NOT_READY runtime must not call the extractor or mutate durable work.
        require_session_service_ready(self.service)
        committed = failed = dead_letter = recovered = 0
        released: list[str] = []
        self.service.commit_group_store.recover_expired_consumers()
        for group in self.service.commit_group_store.pending()[:batch_size]:
            require_session_service_ready(self.service)
            try:
                archive = self.service.archive_store.read_archive_at_manifest(
                    group.archive_uri,
                    group.manifest_digest,
                    tenant_id=group.tenant_id,
                )
                result = self.service.async_commit(archive)
                recovered += int(result.done)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError):
                failed += 1
            except Exception:
                # SessionCommitService terminalizes unclassified canonical
                # attempts; keep the recovery scan moving to queued work.
                failed += 1
            if not session_service_is_ready(self.service):
                return {
                    "claimed": 0,
                    "committed": committed,
                    "failed": failed,
                    "dead_letter": dead_letter,
                    "recovered": recovered,
                    "released": released,
                    "status": "not_ready",
                }
        # A group scan may have changed readiness without raising.  Never
        # lease queue work unless the post-scan state is still exactly READY.
        require_session_service_ready(self.service)
        jobs = self.service.queue_store.lease(
            "session_commit",
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
                if result.done:
                    if not session_service_is_ready(self.service):
                        failed += 1
                        released.extend(self._release_unattempted(jobs[position:]))
                        break
                    self.service.queue_store.ack(job)
                    committed += 1
                else:
                    if not session_service_is_ready(self.service):
                        failed += 1
                        released.extend(self._release_unattempted(jobs[position:]))
                        break
                    retryable = self._result_retryable(result.commit_group_status)
                    status = self._retry(
                        job,
                        RuntimeError(result.status),
                        max_retries=max_retries,
                        retryable=retryable,
                    )
                    dead_letter += int(status == "dead_letter")
                    failed += 1
            except (
                CommitGroupIntegrityError,
                PlanningEnvelopeIntegrityError,
                SalienceLedgerIntegrityError,
                ValueError,
                KeyError,
                TypeError,
            ) as exc:
                if not session_service_is_ready(self.service):
                    failed += 1
                    released.extend(self._release_unattempted(jobs[position:]))
                    break
                status = self._retry(job, exc, max_retries=max_retries, retryable=False)
                dead_letter += int(status == "dead_letter")
                failed += 1
            except (OSError, RuntimeError) as exc:
                if not session_service_is_ready(self.service):
                    failed += 1
                    released.extend(self._release_unattempted(jobs[position:]))
                    break
                status = self._retry(job, exc, max_retries=max_retries, retryable=True)
                dead_letter += int(status == "dead_letter")
                failed += 1
            except Exception as exc:
                # Unclassified failures must still surrender their fenced
                # queue lease and are terminal unless explicitly typed above.
                if not session_service_is_ready(self.service):
                    failed += 1
                    released.extend(self._release_unattempted(jobs[position:]))
                    break
                status = self._retry(job, exc, max_retries=max_retries, retryable=False)
                dead_letter += int(status == "dead_letter")
                failed += 1
        summary: dict[str, object] = {
            "claimed": len(jobs),
            "committed": committed,
            "failed": failed,
            "dead_letter": dead_letter,
            "recovered": recovered,
        }
        if released:
            summary["released"] = released
        if not session_service_is_ready(self.service):
            summary["status"] = "not_ready"
        return summary

    def _release_unattempted(self, jobs: list[QueueJob]) -> list[str]:
        """Fenced release for a batch aborted by a global readiness failure."""

        released: list[str] = []
        for job in jobs:
            settled = self.service.queue_store.release(job)
            if (
                settled.status != "pending"
                or settled.retry_count != job.retry_count
                or settled.lease_token
                or settled.lease_owner
            ):
                raise RuntimeError("session commit batch release did not preserve queue state")
            released.append(job.job_id)
        return released

    def _result_retryable(self, payload: dict) -> bool:  # noqa: ANN001
        if not payload:
            return True
        if payload.get("canonical_status") != "completed":
            return bool(payload.get("canonical_retryable", True))
        return any(
            item.get("status") != "completed" and item.get("retryable", True)
            for item in dict(payload.get("consumers", {}) or {}).values()
        )

    def _retry(self, job: QueueJob, exc: Exception, *, max_retries: int, retryable: bool) -> str:
        settled = self.service.queue_store.retry(
            job,
            exc.__class__.__name__,
            max_retries=max_retries,
            retryable=retryable,
        )
        return settled.status
