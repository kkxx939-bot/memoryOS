"""ContextDB store protocols with lazy historical adapter exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memoryos.adapters.persistence.filesystem.source_store import FileSystemSourceStore as FileSystemSourceStore
    from memoryos.adapters.persistence.in_memory import (
        InMemoryIndexStore as InMemoryIndexStore,
    )
    from memoryos.adapters.persistence.in_memory import (
        InMemoryLockStore as InMemoryLockStore,
    )
    from memoryos.adapters.persistence.in_memory import (
        InMemoryQueueStore as InMemoryQueueStore,
    )
    from memoryos.adapters.persistence.in_memory import (
        InMemoryRelationStore as InMemoryRelationStore,
    )
    from memoryos.adapters.persistence.sqlite import (
        SQLiteIndexStore as SQLiteIndexStore,
    )
    from memoryos.adapters.persistence.sqlite import (
        SqliteIndexStore as SqliteIndexStore,
    )
    from memoryos.adapters.persistence.sqlite import (
        SQLiteLockStore as SQLiteLockStore,
    )
    from memoryos.adapters.persistence.sqlite import (
        SqliteLockStore as SqliteLockStore,
    )
    from memoryos.adapters.persistence.sqlite import (
        SQLiteQueueStore as SQLiteQueueStore,
    )
    from memoryos.adapters.persistence.sqlite import (
        SqliteQueueStore as SqliteQueueStore,
    )
    from memoryos.adapters.persistence.sqlite import (
        SQLiteRelationStore as SQLiteRelationStore,
    )
    from memoryos.adapters.persistence.sqlite import (
        SqliteRelationStore as SqliteRelationStore,
    )
    from memoryos.contextdb.store.index_consistency import (
        IndexConsistencyResult as IndexConsistencyResult,
    )
    from memoryos.contextdb.store.index_consistency import (
        IndexConsistencyService as IndexConsistencyService,
    )
    from memoryos.contextdb.store.index_store import CatalogStore as CatalogStore
    from memoryos.contextdb.store.index_store import IndexHit as IndexHit
    from memoryos.contextdb.store.index_store import IndexStore as IndexStore
    from memoryos.contextdb.store.index_store import (
        MemoryDocumentProjectionStore as MemoryDocumentProjectionStore,
    )
    from memoryos.contextdb.store.lock_store import LockStore as LockStore
    from memoryos.contextdb.store.lock_store import LockToken as LockToken
    from memoryos.contextdb.store.queue_store import QueueJob as QueueJob
    from memoryos.contextdb.store.queue_store import QueueStore as QueueStore
    from memoryos.contextdb.store.relation_store import RelationStore as RelationStore
    from memoryos.contextdb.store.source_store import SourceStore as SourceStore

_EXPORTS = {
    "FileSystemSourceStore": ("memoryos.adapters.persistence.filesystem.source_store", "FileSystemSourceStore"),
    "IndexConsistencyResult": ("memoryos.contextdb.store.index_consistency", "IndexConsistencyResult"),
    "IndexConsistencyService": ("memoryos.contextdb.store.index_consistency", "IndexConsistencyService"),
    "CatalogStore": ("memoryos.contextdb.store.index_store", "CatalogStore"),
    "IndexHit": ("memoryos.contextdb.store.index_store", "IndexHit"),
    "IndexStore": ("memoryos.contextdb.store.index_store", "IndexStore"),
    "MemoryDocumentProjectionStore": (
        "memoryos.contextdb.store.index_store",
        "MemoryDocumentProjectionStore",
    ),
    "InMemoryIndexStore": ("memoryos.adapters.persistence.in_memory.index_store", "InMemoryIndexStore"),
    "InMemoryLockStore": ("memoryos.adapters.persistence.in_memory.lock_store", "InMemoryLockStore"),
    "InMemoryQueueStore": ("memoryos.adapters.persistence.in_memory.queue_store", "InMemoryQueueStore"),
    "InMemoryRelationStore": ("memoryos.adapters.persistence.in_memory.relation_store", "InMemoryRelationStore"),
    "LockStore": ("memoryos.contextdb.store.lock_store", "LockStore"),
    "LockToken": ("memoryos.contextdb.store.lock_store", "LockToken"),
    "QueueJob": ("memoryos.contextdb.store.queue_store", "QueueJob"),
    "QueueStore": ("memoryos.contextdb.store.queue_store", "QueueStore"),
    "RelationStore": ("memoryos.contextdb.store.relation_store", "RelationStore"),
    "SourceStore": ("memoryos.contextdb.store.source_store", "SourceStore"),
    "SQLiteIndexStore": ("memoryos.adapters.persistence.sqlite.index_store", "SQLiteIndexStore"),
    "SQLiteLockStore": ("memoryos.adapters.persistence.sqlite.lock_store", "SQLiteLockStore"),
    "SQLiteQueueStore": ("memoryos.adapters.persistence.sqlite.queue_store", "SQLiteQueueStore"),
    "SQLiteRelationStore": ("memoryos.adapters.persistence.sqlite.relation_store", "SQLiteRelationStore"),
    "SqliteIndexStore": ("memoryos.adapters.persistence.sqlite.index_store", "SqliteIndexStore"),
    "SqliteLockStore": ("memoryos.adapters.persistence.sqlite.lock_store", "SqliteLockStore"),
    "SqliteQueueStore": ("memoryos.adapters.persistence.sqlite.queue_store", "SqliteQueueStore"),
    "SqliteRelationStore": ("memoryos.adapters.persistence.sqlite.relation_store", "SqliteRelationStore"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
