from __future__ import annotations

from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryQueueStore
from memoryos.contextdb.store.vector_store import InMemoryVectorStore


class EmbeddingWorker:
    def __init__(self, source_store: FileSystemSourceStore, queue_store: InMemoryQueueStore, vector_store: InMemoryVectorStore) -> None:
        self.source_store = source_store
        self.queue_store = queue_store
        self.vector_store = vector_store

    def process_pending(self, limit: int = 10) -> dict:
        processed = []
        for job in self.queue_store.lease("embedding", limit=limit):
            content = self.source_store.read_content(job.target_uri)
            embedding = self._hash_embedding(content)
            self.vector_store.upsert_vector(job.target_uri, embedding, metadata={"job_id": job.job_id})
            self.queue_store.ack(job.job_id)
            processed.append(job.job_id)
        return {"processed": processed}

    def _hash_embedding(self, text: str) -> list[float]:
        buckets = [0.0] * 16
        for index, char in enumerate(text):
            buckets[index % len(buckets)] += (ord(char) % 31) / 31.0
        norm = sum(value * value for value in buckets) ** 0.5 or 1.0
        return [round(value / norm, 6) for value in buckets]
