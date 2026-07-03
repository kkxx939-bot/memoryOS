from __future__ import annotations

from memoryos.ports.repositories.memory_repository import MemoryRepository


class ReindexWorker:
    def __init__(self, store: MemoryRepository) -> None:
        self.store = store

    def reindex(self, user_id: str | None = None) -> dict:
        self.store.reindex(user_id=user_id)
        return {"status": "reindexed", "user_id": user_id}
