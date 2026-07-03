from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IndexState:
    source_id: str
    content_hash: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    vector_index_backend: str
    embedding_version: str = "v1"
    index_status: str = "pending"
    indexed_at: str = ""
    last_index_error: str = ""

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "content_hash": self.content_hash,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "vector_index_backend": self.vector_index_backend,
            "embedding_version": self.embedding_version,
            "index_status": self.index_status,
            "indexed_at": self.indexed_at,
            "last_index_error": self.last_index_error,
        }


class IndexConsistencyService:
    def is_stale(self, current: IndexState | None, desired: IndexState) -> bool:
        if current is None:
            return True
        return any(
            [
                current.content_hash != desired.content_hash,
                current.embedding_provider != desired.embedding_provider,
                current.embedding_model != desired.embedding_model,
                current.embedding_dimension != desired.embedding_dimension,
                current.vector_index_backend != desired.vector_index_backend,
                current.embedding_version != desired.embedding_version,
                current.index_status != "indexed",
            ]
        )
