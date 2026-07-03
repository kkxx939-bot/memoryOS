from __future__ import annotations

from typing import Any, Protocol


class _MemoryLifecycleStore(Protocol):
    def lifecycle_report(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]: ...
    def archive_cold_memories(
        self,
        user_id: str,
        limit: int = 20,
        max_hotness: float = 0.12,
        allowed_types: set[str] | None = None,
    ) -> dict[str, Any]: ...
    def reindex(self, user_id: str | None = None) -> None: ...
    def verify_index(self, user_id: str | None = None) -> dict[str, Any]: ...


class SqliteMemoryLifecycleRepository:
    def __init__(self, store: _MemoryLifecycleStore) -> None:
        self.store = store

    def lifecycle_report(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        return self.store.lifecycle_report(user_id, limit=limit)

    def archive_cold_memories(
        self,
        user_id: str,
        limit: int = 20,
        max_hotness: float = 0.12,
        allowed_types: set[str] | None = None,
    ) -> dict[str, Any]:
        return self.store.archive_cold_memories(
            user_id=user_id,
            limit=limit,
            max_hotness=max_hotness,
            allowed_types=allowed_types,
        )

    def reindex(self, user_id: str | None = None) -> None:
        self.store.reindex(user_id=user_id)

    def verify_index(self, user_id: str | None = None) -> dict[str, Any]:
        return self.store.verify_index(user_id=user_id)
