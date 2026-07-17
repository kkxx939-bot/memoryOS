"""Compatibility exports for persistence adapters moved out of ContextDB."""

from memoryos.adapters.persistence.filesystem import BundleIntegrityError, FileSystemSourceStore
from memoryos.adapters.persistence.in_memory import (
    InMemoryIndexStore,
    InMemoryLockStore,
    InMemoryQueueStore,
    InMemoryRelationStore,
)

__all__ = [
    "BundleIntegrityError",
    "FileSystemSourceStore",
    "InMemoryIndexStore",
    "InMemoryLockStore",
    "InMemoryQueueStore",
    "InMemoryRelationStore",
]
