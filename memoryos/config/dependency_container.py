from __future__ import annotations

from memoryos.adapters.persistence.sqlite.sqlite_memory_repository import MemoryStore
from memoryos.config.provider_registry import ProviderRegistry
from memoryos.config.settings import Settings, load_settings
from memoryos.ports.repositories.memory_repository import MemoryRepository


def build_memory_store(settings: Settings | None = None) -> MemoryRepository:
    current = settings or load_settings()
    registry = ProviderRegistry(current)
    return MemoryStore(
        current.memory_root,
        embedding_provider=registry.get_embedding_provider(),
        rerank_provider=registry.get_rerank_provider(),
    )


def build_provider_registry(settings: Settings | None = None) -> ProviderRegistry:
    return ProviderRegistry(settings or load_settings())
