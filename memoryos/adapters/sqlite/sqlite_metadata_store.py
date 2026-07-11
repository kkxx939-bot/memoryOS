"""适配器里的SQLite元数据存储。"""

from memoryos.adapters.sqlite.sqlite_index_store import SqliteIndexStore

SqliteMetadataStore = SqliteIndexStore

__all__ = ["SqliteMetadataStore"]
