"""Compatibility alias for the SQLite catalog adapter."""

from memoryos.adapters.persistence.sqlite.index_store import SqliteIndexStore

SqliteMetadataStore = SqliteIndexStore

__all__ = ["SqliteMetadataStore"]
