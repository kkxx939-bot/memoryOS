"""后台任务里的记忆提案任务。"""

from __future__ import annotations

from memoryos.contextdb.session.session_commit import SessionCommitService


class MemoryProposalWorker:
    """跑 MemoryProposalWorker 对应的后台任务。"""

    def __init__(self, service: SessionCommitService) -> None:
        self.service = service

    def process_pending(self, *, batch_size: int = 10, lease_seconds: int = 60, max_retries: int = 3) -> dict:
        committed = failed = dead_letter = 0
        jobs = self.service.queue_store.lease("memory_proposal", limit=batch_size, lease_seconds=lease_seconds)
        for job in jobs:
            try:
                archive = self.service.archive_store.read_archive(
                    job.target_uri,
                    tenant_id=str(job.payload.get("tenant_id") or "default"),
                    manifest_digest=str(job.payload.get("manifest_digest") or "") or None,
                )
                result = self.service.async_commit(archive)
                if result.canonical_committed:
                    self.service.queue_store.ack(job.job_id)
                    committed += 1
                    continue
                raise RuntimeError(result.status)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                retry = getattr(self.service.queue_store, "retry", None)
                if callable(retry):
                    status = str(
                        retry(
                            job.job_id,
                            type(exc).__name__,
                            max_retries=max_retries,
                            retryable=True,
                        )
                    )
                else:
                    self.service.queue_store.fail(job.job_id, type(exc).__name__)
                    status = "failed"
                failed += 1
                dead_letter += int(status == "dead_letter")
        return {"claimed": len(jobs), "committed": committed, "failed": failed, "dead_letter": dead_letter}
