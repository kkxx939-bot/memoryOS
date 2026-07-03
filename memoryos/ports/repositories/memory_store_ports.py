from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from memoryos.domain.memory.memory_item import MemoryItem
from memoryos.ports.indexes.vector_index import VectorHit, VectorRecord
from memoryos.ports.repositories.index_job_repository import IndexJob


@runtime_checkable
class SourceStore(Protocol):
    """Durable memory facts. This is the source of truth."""

    def upsert_memory(self, item: MemoryItem) -> dict[str, Any]: ...

    def patch_memory(self, identifier: str, user_id: str, patch: dict[str, Any]) -> dict[str, Any]: ...

    def mark_obsolete(self, identifier: str, user_id: str, superseded_by: str) -> dict[str, Any]: ...

    def soft_delete_memory(self, identifier: str, user_id: str, reason: str) -> dict[str, Any]: ...


@runtime_checkable
class IndexStore(Protocol):
    """Keyword/BM25/FTS derived index. It must be rebuildable from SourceStore."""

    def upsert_text(self, memory: dict[str, Any], content: str) -> None: ...

    def delete_text(self, memory_id: str, user_id: str) -> None: ...

    def search_text(self, query: str, user_id: str, limit: int) -> list[dict[str, Any]]: ...

    def rebuild_text_index(self, user_id: str | None = None) -> None: ...


@runtime_checkable
class MemoryVectorStore(Protocol):
    """Vector index derived from source facts and embedding jobs."""

    def upsert_vector(self, record: VectorRecord) -> None: ...

    def delete_vector(self, namespace: str, vector_id: str) -> None: ...

    def search_vector(self, namespace: str, query_vector: list[float], top_k: int) -> list[VectorHit]: ...

    def rebuild_vector_index(self, user_id: str | None = None) -> None: ...


@runtime_checkable
class JobStore(Protocol):
    """Durable background jobs for indexing, redo, reindex, and recovery."""

    def enqueue(self, job: IndexJob) -> None: ...

    def claim(self, worker_id: str, limit: int = 10) -> list[IndexJob]: ...

    def mark_done(self, job_id: str) -> None: ...

    def mark_failed(self, job_id: str, error: str) -> None: ...
