"""后台任务里的记忆提案任务。"""

from __future__ import annotations

import os
import uuid

from memoryos.contextdb.session.session_commit import SessionCommitService


class MemoryProposalWorker:
    """跑 MemoryProposalWorker 对应的后台任务。"""

    def __init__(self, service: SessionCommitService, *, worker_id: str | None = None) -> None:
        self.service = service
        self.worker_id = worker_id or f"memory-proposal:{os.getpid()}:{uuid.uuid4().hex}"

    def process_pending(self, *, batch_size: int = 10, lease_seconds: int = 60, max_retries: int = 3) -> dict:
        committed = failed = dead_letter = 0
        jobs = self.service.queue_store.lease(
            "memory_proposal",
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
                if result.canonical_committed:
                    self.service.queue_store.ack(job)
                    committed += 1
                    continue
                raise RuntimeError(result.status)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                status = self.service.queue_store.retry(
                    job,
                    type(exc).__name__,
                    max_retries=max_retries,
                    retryable=True,
                ).status
                failed += 1
                dead_letter += int(status == "dead_letter")
        return {"claimed": len(jobs), "committed": committed, "failed": failed, "dead_letter": dead_letter}
