from __future__ import annotations

from memoryos.application.retrieval.memory_context_builder import MemoryContextBuilder
from memoryos.infrastructure.repositories.memory_repository import MemoryStore


class MemoryHook:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def build_digest(self, user_id: str, query: str, limit: int = 6) -> str:
        context = MemoryContextBuilder(self.store).build(user_id=user_id, query=query, digest_limit=limit)
        return context.digest
