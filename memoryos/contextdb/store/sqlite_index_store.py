"""Compatibility exports for the SQLite catalog adapter."""

from memoryos.adapters.persistence.sqlite.index_store import (
    SQLiteIndexStore,
    SqliteIndexStore,
    lexical_match_count,
    lexical_relevance,
    lexical_terms,
)

__all__ = [
    "SQLiteIndexStore",
    "SqliteIndexStore",
    "lexical_match_count",
    "lexical_relevance",
    "lexical_terms",
]
