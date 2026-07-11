"""适配器里的文件系统锁存储。"""

from memoryos.contextdb.store.local_stores import InMemoryLockStore

FileSystemLockStore = InMemoryLockStore

__all__ = ["FileSystemLockStore"]
