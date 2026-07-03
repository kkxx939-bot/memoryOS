from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

INDEX_JOB_STATES = {
    "pending",
    "embedding",
    "indexed",
    "failed",
    "stale",
    "delete_pending",
    "deleted",
}


@dataclass(frozen=True)
class IndexJob:
    job_id: str
    user_id: str
    source_type: str
    source_id: str
    operation: str
    status: str = "pending"
    namespace: str = ""
    content_hash: str = ""
    embedding_provider: str = ""
    embedding_model: str = ""
    embedding_dimension: int = 0
    vector_index_backend: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "user_id": self.user_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "operation": self.operation,
            "status": self.status,
            "namespace": self.namespace,
            "content_hash": self.content_hash,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "vector_index_backend": self.vector_index_backend,
            "metadata": self.metadata,
        }


class IndexJobRepository(Protocol):
    def enqueue(self, job: IndexJob) -> dict: ...

    def pending(self, user_id: str | None = None, limit: int = 50) -> list[dict]: ...

    def mark(self, job_id: str, status: str, patch: dict | None = None) -> dict: ...
