from __future__ import annotations

from memoryos.infrastructure.repositories.memory_repository import MemoryStore


class MemoryIndexRepository:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def reindex(self, user_id: str | None = None) -> None:
        self.store.reindex(user_id=user_id)
