"""基于锁存储协议协调同一路径上的写操作。"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from infrastructure.store.contracts.lock import LockStore, LockToken


@dataclass(frozen=True)
class LeaseGuard:
    lock_store: LockStore
    token: LockToken
    ttl_seconds: int

    def checkpoint(self) -> None:
        self.lock_store.renew(self.token, ttl_seconds=self.ttl_seconds)

    @contextmanager
    def fenced(self) -> Iterator[None]:
        with self.lock_store.fenced((self.token,), ttl_seconds=self.ttl_seconds):
            yield


class PathLock:
    def __init__(self, lock_store: LockStore) -> None:
        self.lock_store = lock_store

    @contextmanager
    def acquire(self, key: str, ttl_seconds: int = 30) -> Iterator[LeaseGuard]:
        token = self.lock_store.acquire(key, ttl_seconds=ttl_seconds)
        try:
            yield LeaseGuard(self.lock_store, token, max(1, ttl_seconds))
        finally:
            self.lock_store.release(token)

    @contextmanager
    def fenced(self, guards: Sequence[LeaseGuard]) -> Iterator[None]:
        if not guards:
            yield
            return
        if any(guard.lock_store is not self.lock_store for guard in guards):
            raise ValueError("all lease guards must belong to one LockStore")
        ttl_seconds = min(guard.ttl_seconds for guard in guards)
        with self.lock_store.fenced(
            tuple(guard.token for guard in guards),
            ttl_seconds=ttl_seconds,
        ):
            yield
