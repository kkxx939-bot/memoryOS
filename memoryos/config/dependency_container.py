from __future__ import annotations

from memoryos.adapters.persistence.sqlite.sqlite_memory_repository import MemoryStore
from memoryos.config.settings import Settings, load_settings
from memoryos.ports.repositories.memory_repository import MemoryRepository


def build_memory_store(settings: Settings | None = None) -> MemoryRepository:
    current = settings or load_settings()
    return MemoryStore(current.memory_root)
