from __future__ import annotations

from memoryos.ports.repositories.memory_repository import MemoryRepository
from memoryos.services.retrieval.memory_context_builder import MemoryContextBuilder


class MemoryHook:
    def __init__(self, store: MemoryRepository) -> None:
        self.store = store

    def build_digest(self, user_id: str, query: str, limit: int = 6) -> str:
        context = MemoryContextBuilder(self.store).build(user_id=user_id, query=query, digest_limit=limit)
        return context.digest
