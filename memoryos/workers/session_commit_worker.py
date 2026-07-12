"""后台任务里的会话提交任务。"""

from __future__ import annotations

import os
import uuid

from memoryos.contextdb.session.session_commit import SessionCommitService
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.source_store import QueueJob


class SessionCommitWorker:
    def __init__(self, service: SessionCommitService, *, worker_id: str | None = None) -> None:
        self.service = service
        self.worker_id = worker_id or f"session-commit:{os.getpid()}:{uuid.uuid4().hex}"

    def process_archive(self, archive: SessionArchive) -> dict:
        result = self.service.async_commit(archive)
        return {"task_id": result.task_id, "status": result.status, "done": result.done}

    def process_pending(self, *, batch_size: int = 10, lease_seconds: int = 60, max_retries: int = 3) -> dict:
        committed = failed = dead_letter = recovered = 0
        self.service.commit_group_store.recover_expired_consumers()
        for group in self.service.commit_group_store.pending()[:batch_size]:
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
        jobs = self.service.queue_store.lease(
            "session_commit",
            lease_owner=self.worker_id,
            limit=batch_size,
            lease_seconds=lease_seconds,
        )
        for job in jobs:
            try:
                archive = self.service.archive_store.read_archive(
                    job.target_uri,
                    tenant_id=str(job.payload.get("tenant_id") or "default"),
                    manifest_digest=str(job.payload.get("manifest_digest") or "") or None,
                )
                result = self.service.async_commit(archive)
                if result.done:
                    self.service.queue_store.ack(job)
                    committed += 1
                else:
                    retryable = self._result_retryable(result.commit_group_status)
                    status = self._retry(
                        job,
                        RuntimeError(result.status),
                        max_retries=max_retries,
                        retryable=retryable,
                    )
                    dead_letter += int(status == "dead_letter")
                    failed += 1
            except (ValueError, KeyError, TypeError) as exc:
                status = self._retry(job, exc, max_retries=max_retries, retryable=False)
                dead_letter += int(status == "dead_letter")
                failed += 1
            except (OSError, RuntimeError) as exc:
                status = self._retry(job, exc, max_retries=max_retries, retryable=True)
                dead_letter += int(status == "dead_letter")
                failed += 1
        return {
            "claimed": len(jobs),
            "committed": committed,
            "failed": failed,
            "dead_letter": dead_letter,
            "recovered": recovered,
        }

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
