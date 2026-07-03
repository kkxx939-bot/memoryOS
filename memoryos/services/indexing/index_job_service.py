from __future__ import annotations

import hashlib

from memoryos.ports.providers.embedding_provider import EmbeddingProvider
from memoryos.ports.repositories.index_job_repository import IndexJob, IndexJobRepository
from memoryos.services.indexing.chunking_service import MemoryChunk


class IndexJobService:
    def __init__(
        self,
        jobs: IndexJobRepository,
        embedding_provider: EmbeddingProvider,
        vector_index_backend: str = "sqlite",
    ) -> None:
        self.jobs = jobs
        self.embedding_provider = embedding_provider
        self.vector_index_backend = vector_index_backend

    def enqueue_upsert(self, user_id: str, chunk: MemoryChunk, namespace: str = "memory") -> dict:
        result = self.embedding_provider.embed_text(chunk.text)
        job = IndexJob(
            job_id=self._job_id(user_id, chunk.chunk_id, "upsert", result.model),
            user_id=user_id,
            source_type=chunk.source_type,
            source_id=chunk.source_id,
            operation="upsert",
            namespace=namespace,
            content_hash=chunk.content_hash,
            embedding_provider=result.provider,
            embedding_model=result.model,
            embedding_dimension=result.dimension,
            vector_index_backend=self.vector_index_backend,
            metadata={"chunk": chunk.to_dict(), "embedding": result.to_dict()},
        )
        return self.jobs.enqueue(job)

    def enqueue_delete(self, user_id: str, source_type: str, source_id: str, namespace: str = "memory") -> dict:
        job = IndexJob(
            job_id=self._job_id(user_id, source_id, "delete", namespace),
            user_id=user_id,
            source_type=source_type,
            source_id=source_id,
            operation="delete",
            status="delete_pending",
            namespace=namespace,
            vector_index_backend=self.vector_index_backend,
        )
        return self.jobs.enqueue(job)

    def _job_id(self, user_id: str, source_id: str, operation: str, version: str) -> str:
        material = f"{user_id}:{source_id}:{operation}:{version}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
