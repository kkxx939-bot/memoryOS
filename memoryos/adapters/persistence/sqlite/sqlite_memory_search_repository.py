from __future__ import annotations

from typing import Any, Protocol


class _MemorySearchStore(Protocol):
    def search(
        self,
        query: str,
        user_id: str,
        memory_type: str | None = None,
        limit: int = 5,
        touch: bool = True,
    ) -> list[dict[str, Any]]: ...
    def hybrid_search(
        self,
        query: str,
        user_id: str,
        memory_type: str | None = None,
        limit: int = 8,
        touch: bool = False,
    ) -> list[dict[str, Any]]: ...
    def list_by_type(
        self,
        user_id: str,
        memory_type: str,
        limit: int = 8,
        touch: bool = False,
    ) -> list[dict[str, Any]]: ...


class SqliteMemorySearchRepository:
    def __init__(self, store: _MemorySearchStore) -> None:
        self.store = store

    def search(
        self,
        query: str,
        user_id: str,
        memory_type: str | None = None,
        limit: int = 5,
        touch: bool = True,
    ) -> list[dict[str, Any]]:
        return self.store.search(query, user_id=user_id, memory_type=memory_type, limit=limit, touch=touch)

    def hybrid_search(
        self,
        query: str,
        user_id: str,
        memory_type: str | None = None,
        limit: int = 8,
        touch: bool = False,
    ) -> list[dict[str, Any]]:
        return self.store.hybrid_search(query, user_id=user_id, memory_type=memory_type, limit=limit, touch=touch)

    def list_by_type(
        self,
        user_id: str,
        memory_type: str,
        limit: int = 8,
        touch: bool = False,
    ) -> list[dict[str, Any]]:
        return self.store.list_by_type(user_id, memory_type, limit=limit, touch=touch)
