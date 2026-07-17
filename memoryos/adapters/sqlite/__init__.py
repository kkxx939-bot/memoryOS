"""Compatibility exports for the canonical persistence adapter package."""

from memoryos.adapters.persistence.sqlite import (
    SQLiteIndexStore,
    SqliteIndexStore,
    SQLiteQueueStore,
    SqliteQueueStore,
    SQLiteRelationStore,
    SqliteRelationStore,
)

SqliteMetadataStore = SqliteIndexStore

__all__ = [
    "SQLiteIndexStore",
    "SQLiteQueueStore",
    "SQLiteRelationStore",
    "SqliteIndexStore",
    "SqliteMetadataStore",
    "SqliteQueueStore",
    "SqliteRelationStore",
]
