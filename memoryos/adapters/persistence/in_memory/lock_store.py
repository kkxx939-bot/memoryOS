"""In-memory LockStore adapter."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from memoryos.contextdb.store.lock_store import LockLostError, LockToken


class InMemoryLockStore:
    def __init__(self) -> None:
        self.locks: dict[str, tuple[str, int, datetime]] = {}
        self.fences: dict[str, int] = {}
        self._guard = threading.RLock()

    def acquire(self, lock_key: str, ttl_seconds: int = 30) -> LockToken:
        with self._guard:
            now = datetime.now(timezone.utc)
            existing = self.locks.get(lock_key)
            if existing is not None and existing[2] > now:
                raise TimeoutError(f"Lock already held: {lock_key}")
            fence = self.fences.get(lock_key, 0) + 1
            self.fences[lock_key] = fence
            token = uuid.uuid4().hex
            self.locks[lock_key] = (
                token,
                fence,
                now + timedelta(seconds=max(1, ttl_seconds)),
            )
            return LockToken(lock_key=lock_key, token=token, fence=fence)

    def renew(self, token: LockToken, ttl_seconds: int = 30) -> LockToken:
        with self._guard:
            self._assert_owned_unlocked(token)
            self.locks[token.lock_key] = (
                token.token,
                token.fence,
                datetime.now(timezone.utc) + timedelta(seconds=max(1, ttl_seconds)),
            )
        return token

    def assert_owned(self, token: LockToken) -> None:
        with self._guard:
            self._assert_owned_unlocked(token)

    @contextmanager
    def fenced(self, tokens: Sequence[LockToken], ttl_seconds: int = 30) -> Iterator[None]:
        with self._guard:
            for token in tokens:
                self._assert_owned_unlocked(token)
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=max(1, ttl_seconds))
            for token in tokens:
                self.locks[token.lock_key] = (token.token, token.fence, expires_at)
            yield
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=max(1, ttl_seconds))
            for token in tokens:
                self._assert_identity_unlocked(token)
                self.locks[token.lock_key] = (token.token, token.fence, expires_at)

    def release(self, token: LockToken) -> None:
        with self._guard:
            current = self.locks.get(token.lock_key)
            if current is not None and current[:2] == (token.token, token.fence):
                self.locks.pop(token.lock_key, None)

    def _assert_owned_unlocked(self, token: LockToken) -> None:
        self._assert_identity_unlocked(token)
        current = self.locks[token.lock_key]
        if current[2] <= datetime.now(timezone.utc):
            raise LockLostError(f"Lock lease lost: {token.lock_key}")
    def _assert_identity_unlocked(self, token: LockToken) -> None:
        current = self.locks.get(token.lock_key)
        if current is None or current[:2] != (token.token, token.fence):
            raise LockLostError(f"Lock lease lost: {token.lock_key}")
