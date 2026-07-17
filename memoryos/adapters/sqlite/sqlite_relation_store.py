"""Compatibility exports for the canonical SQLite relation adapter path."""

from memoryos.adapters.persistence.sqlite.relation_store import (
    SQLiteRelationStore,
    SqliteRelationStore,
)

__all__ = ["SQLiteRelationStore", "SqliteRelationStore"]
