from __future__ import annotations

from memoryos.config.dependency_container import build_memory_store
from memoryos.config.settings import Settings, load_settings
from memoryos.ports.repositories.memory_repository import MemoryRepository


def build_store(settings: Settings | None = None) -> MemoryRepository:
    return build_memory_store(settings or load_settings())
