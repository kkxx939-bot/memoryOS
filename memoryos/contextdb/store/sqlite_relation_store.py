"""Compatibility exports for the SQLite relation adapter."""

from memoryos.adapters.persistence.sqlite.relation_store import (
    SQLiteRelationStore,
    SqliteRelationStore,
)

__all__ = ["SQLiteRelationStore", "SqliteRelationStore"]
