"""后台任务里的语义任务。"""

from __future__ import annotations

import os
import uuid

from memoryos.contextdb.layers.layer_refresher import LayerRefresher
from memoryos.contextdb.store.source_store import QueueStore, SourceStore


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
        processed = []
        failed: list[str] = []
        dead_letter: list[str] = []
        jobs = self.queue_store.lease(
            "semantic",
            lease_owner=self.worker_id,
            limit=limit,
            lease_seconds=lease_seconds,
        )
        for job in jobs:
            try:
                obj = self.source_store.read_object(job.target_uri)
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
        }
