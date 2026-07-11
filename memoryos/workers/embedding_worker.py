"""后台任务里的向量化任务。"""

from __future__ import annotations

from collections.abc import Callable

from memoryos.contextdb.store.source_store import QueueStore, SourceStore
from memoryos.contextdb.store.vector_store import VectorStore
from memoryos.providers.embedding import EmbeddingProvider, HashingEmbeddingProvider


class EmbeddingWorker:
    def __init__(
        self,
        source_store: SourceStore,
        queue_store: QueueStore,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider | None = None,
        namespace_builder: Callable[[str], str] | None = None,
    ) -> None:
        self.source_store = source_store
        self.queue_store = queue_store
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider or HashingEmbeddingProvider()
        self.namespace_builder = namespace_builder

    def process_pending(self, limit: int = 10) -> dict:
        processed = []
        for job in self.queue_store.lease("embedding", limit=limit):
            try:
                content = self.source_store.read_content(job.target_uri)
                embedding = self.embedding_provider.embed(content)
                metadata = {
                    "job_id": job.job_id,
                    "embedding_model": self.embedding_provider.model_name,
                    "embedding_dimension": self.embedding_provider.dimension,
                    "source_uri": job.target_uri,
                    "schema_version": "vector_embedding_v1",
                }
                if self.namespace_builder is not None:
                    metadata["namespace"] = self.namespace_builder(job.target_uri)
                self.vector_store.upsert_vector(job.target_uri, embedding, metadata=metadata)
            except Exception as exc:
                self.queue_store.fail(job.job_id, str(exc))
                continue
            self.queue_store.ack(job.job_id)
            processed.append(job.job_id)
        return {"processed": processed}
