"""上下文数据库里的路径锁。"""

from __future__ import annotations

from contextlib import contextmanager

from memoryos.contextdb.store.source_store import LockStore


class PathLock:
    def __init__(self, lock_store: LockStore) -> None:
        self.lock_store = lock_store

    @contextmanager
    def acquire(self, key: str, ttl_seconds: int = 30):
        token = self.lock_store.acquire(key, ttl_seconds=ttl_seconds)
        try:
            yield token
        finally:
            self.lock_store.release(token)
