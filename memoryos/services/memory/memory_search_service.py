from __future__ import annotations

from memoryos.ports.repositories.memory_repository import MemoryRepository


class MemorySearchService:
    def __init__(self, store: MemoryRepository) -> None:
        self.store = store

    def search(self, user_id: str, query: str, memory_type: str | None = None, limit: int = 8) -> list[dict]:
        return self.store.search(query, user_id=user_id, memory_type=memory_type, limit=limit)

    def hybrid_search(self, user_id: str, query: str, memory_type: str | None = None, limit: int = 8) -> list[dict]:
        return self.store.hybrid_search(query, user_id=user_id, memory_type=memory_type, limit=limit)
