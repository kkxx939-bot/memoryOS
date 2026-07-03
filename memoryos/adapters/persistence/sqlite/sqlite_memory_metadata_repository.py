from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from memoryos.domain.memory.memory_item import MemoryItem


class _MemoryMetadataStore(Protocol):
    def add_memory(self, item: MemoryItem) -> Path: ...
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
    def resolve_memory(self, identifier: str, user_id: str) -> dict[str, Any]: ...
    def merge_memory(self, target_identifier: str, source_identifier: str, user_id: str) -> dict[str, Any]: ...


class SqliteMemoryMetadataRepository:
    def __init__(self, store: _MemoryMetadataStore) -> None:
        self.store = store

    def add(self, item: MemoryItem) -> Path:
        return self.store.add_memory(item)

    def update(
        self,
        identifier: str,
        user_id: str,
        title: str | None = None,
        text: str | None = None,
        tags: list[str] | None = None,
        metadata_patch: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.store.update_memory(
            identifier,
            user_id=user_id,
            title=title,
            text=text,
            tags=tags,
            metadata_patch=metadata_patch,
        )

    def delete(self, identifier: str, user_id: str) -> dict[str, Any]:
        return self.store.delete_memory(identifier, user_id)

    def resolve(self, identifier: str, user_id: str) -> dict[str, Any]:
        return self.store.resolve_memory(identifier, user_id)

    def merge(self, target_identifier: str, source_identifier: str, user_id: str) -> dict[str, Any]:
        return self.store.merge_memory(target_identifier, source_identifier, user_id)
