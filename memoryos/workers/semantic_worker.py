from __future__ import annotations

from memoryos.contextdb.layers.layer_refresher import LayerRefresher
from memoryos.contextdb.store.source_store import QueueStore, SourceStore


class SemanticWorker:
    def __init__(self, source_store: SourceStore, queue_store: QueueStore) -> None:
        self.source_store = source_store
        self.queue_store = queue_store

    def process_pending(self, limit: int = 10) -> dict:
        processed = []
        for job in self.queue_store.lease("semantic", limit=limit):
            try:
                obj = self.source_store.read_object(job.target_uri)
                content = self.source_store.read_content(job.target_uri)
                LayerRefresher(self.source_store).refresh(obj, content)
            except Exception as exc:
                self.queue_store.fail(job.job_id, str(exc))
                continue
            self.queue_store.ack(job.job_id)
            processed.append(job.job_id)
        return {"processed": processed}
