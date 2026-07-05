from __future__ import annotations

from typing import Protocol

from memoryos.memory.model.memory import Memory


class MemoryStore(Protocol):
    """Storage boundary for memory-specific repositories."""

    def get_memory(self, uri: str) -> Memory | None: ...

    def upsert_memory(self, memory: Memory) -> None: ...

    def search_memories(self, query: str, *, user_id: str | None = None, limit: int = 10) -> list[Memory]: ...
