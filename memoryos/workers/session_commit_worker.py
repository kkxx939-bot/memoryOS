"""后台任务里的会话提交任务。"""

from __future__ import annotations

from memoryos.contextdb.session.session_commit import SessionCommitService
from memoryos.contextdb.session.session_model import SessionArchive


class SessionCommitWorker:
    def __init__(self, service: SessionCommitService) -> None:
        self.service = service

    def process_archive(self, archive: SessionArchive) -> dict:
        result = self.service.async_commit(archive)
        return {"task_id": result.task_id, "status": result.status, "done": result.done}

    def process_pending(self, *, batch_size: int = 10, lease_seconds: int = 60, max_retries: int = 3) -> dict:
        committed = failed = dead_letter = 0
        jobs = self.service.queue_store.lease("session_commit", limit=batch_size, lease_seconds=lease_seconds)
        for job in jobs:
            try:
                archive = self.service.archive_store.read_archive(job.target_uri)
                self.process_archive(archive)
                self.service.queue_store.ack(job.job_id)
                committed += 1
            except (ValueError, KeyError, TypeError) as exc:
                status = self._retry(job.job_id, exc, max_retries=max_retries, retryable=False)
                dead_letter += int(status == "dead_letter")
                failed += 1
            except (OSError, RuntimeError) as exc:
                status = self._retry(job.job_id, exc, max_retries=max_retries, retryable=True)
                dead_letter += int(status == "dead_letter")
                failed += 1
        return {"claimed": len(jobs), "committed": committed, "failed": failed, "dead_letter": dead_letter}

    def _retry(self, job_id: str, exc: Exception, *, max_retries: int, retryable: bool) -> str:
        retry = getattr(self.service.queue_store, "retry", None)
        if callable(retry):
            return str(retry(job_id, exc.__class__.__name__, max_retries=max_retries, retryable=retryable))
        self.service.queue_store.fail(job_id, exc.__class__.__name__)
        return "failed"
