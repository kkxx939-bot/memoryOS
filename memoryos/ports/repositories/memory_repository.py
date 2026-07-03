from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from memoryos.domain.memory.memory_item import MemoryItem


@runtime_checkable
class MemoryRepository(Protocol):
    root: Path
    rerank_provider: Any

    def init(self, user_id: str) -> None: ...

    def add_memory(self, item: MemoryItem) -> Path: ...

    def upsert_profile(self, user_id: str, text: str, mode: str = "append") -> dict[str, Any]: ...

    def update_daily_behavior(
        self,
        user_id: str,
        text: str,
        day: str | None = None,
        mode: str = "append",
    ) -> dict[str, Any]: ...

    def record_event(
        self,
        user_id: str,
        event_type: str,
        text: str,
        day: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]: ...

    def resolve_memory(self, identifier: str, user_id: str) -> dict[str, Any]: ...

    def update_memory(
        self,
        identifier: str,
        user_id: str,
        title: str | None = None,
        text: str | None = None,
        tags: list[str] | None = None,
        metadata_patch: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def delete_memory(self, identifier: str, user_id: str) -> dict[str, Any]: ...

    def merge_memory(self, target_identifier: str, source_identifier: str, user_id: str) -> dict[str, Any]: ...

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

    def rank_directory_layers(
        self,
        query: str,
        user_id: str,
        memory_types: set[str] | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]: ...
