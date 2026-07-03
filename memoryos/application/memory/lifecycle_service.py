from __future__ import annotations

from memoryos.infrastructure.repositories.memory_repository import MemoryStore


class MemoryLifecycleService:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def report(self, user_id: str, limit: int = 20) -> list[dict]:
        return self.store.lifecycle_report(user_id=user_id, limit=limit)

    def archive_cold(
        self,
        user_id: str,
        limit: int = 20,
        max_hotness: float = 0.12,
        allowed_types: set[str] | None = None,
    ) -> dict:
        return self.store.archive_cold_memories(
            user_id=user_id,
            limit=limit,
            max_hotness=max_hotness,
            allowed_types=allowed_types,
        )
