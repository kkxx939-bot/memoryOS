"""后台任务里的语义任务。"""

from __future__ import annotations

import os
import uuid

from memoryos.contextdb.layers.layer_refresher import LayerRefresher
from memoryos.contextdb.store.source_store import (
    QueueStore,
    SourceStore,
    is_canonical_memory_object,
    is_canonical_memory_uri,
)
from memoryos.workers.readiness import require_source_store_ready


class SemanticWorker:
    def __init__(
        self,
        source_store: SourceStore,
        queue_store: QueueStore,
        *,
        worker_id: str | None = None,
    ) -> None:
        self.source_store = source_store
        self.queue_store = queue_store
        self.worker_id = worker_id or f"semantic:{os.getpid()}:{uuid.uuid4().hex}"

    def process_pending(
        self,
        limit: int = 10,
        *,
        lease_seconds: int = 60,
        max_retries: int = 3,
    ) -> dict:
        require_source_store_ready(self.source_store)
        processed = []
        failed: list[str] = []
        dead_letter: list[str] = []
        quarantine: list[str] = []
        jobs = self.queue_store.lease(
            "semantic",
            lease_owner=self.worker_id,
            limit=limit,
            lease_seconds=lease_seconds,
        )
        for job in jobs:
            try:
                if is_canonical_memory_uri(job.target_uri):
                    self.queue_store.quarantine(job, "canonical_requires_projector")
                    failed.append(job.job_id)
                    quarantine.append(job.job_id)
                    continue
                obj = self.source_store.read_object(job.target_uri)
                if is_canonical_memory_object(obj):
                    self.queue_store.quarantine(job, "canonical_requires_projector")
                    failed.append(job.job_id)
                    quarantine.append(job.job_id)
                    continue
                content = self.source_store.read_content(job.target_uri)
                LayerRefresher(self.source_store).refresh(obj, content)
            except Exception as exc:
                settled = self.queue_store.retry(
                    job,
                    type(exc).__name__,
                    max_retries=max_retries,
                    retryable=isinstance(exc, OSError),
                )
                failed.append(job.job_id)
                if settled.status == "dead_letter":
                    dead_letter.append(job.job_id)
                continue
            self.queue_store.ack(job)
            processed.append(job.job_id)
        return {
            "claimed": len(jobs),
            "processed": processed,
            "failed": failed,
            "dead_letter": dead_letter,
            "quarantine": quarantine,
        }
