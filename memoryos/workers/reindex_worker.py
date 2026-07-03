from __future__ import annotations

from memoryos.infrastructure.repositories.memory_repository import MemoryStore


class ReindexWorker:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def reindex(self, user_id: str | None = None) -> dict:
        self.store.reindex(user_id=user_id)
        return {"status": "reindexed", "user_id": user_id}
