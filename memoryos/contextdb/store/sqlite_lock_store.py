"""Compatibility exports for the SQLite lock adapter."""

from memoryos.adapters.persistence.sqlite.lock_store import SQLiteLockStore, SqliteLockStore

__all__ = ["SQLiteLockStore", "SqliteLockStore"]
