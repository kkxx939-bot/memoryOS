"""ContextDB lock storage protocol and fencing token."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LockToken:
    lock_key: str
    token: str
    fence: int = 0


class LockLostError(TimeoutError):
    """A writer no longer owns the lease it was issued."""


class LockStore(Protocol):
    def acquire(self, lock_key: str, ttl_seconds: int = 30) -> LockToken: ...

    def renew(self, token: LockToken, ttl_seconds: int = 30) -> LockToken: ...

    def assert_owned(self, token: LockToken) -> None: ...

    def fenced(
        self,
        tokens: Sequence[LockToken],
        ttl_seconds: int = 30,
    ) -> AbstractContextManager[None]: ...

    def release(self, token: LockToken) -> None: ...


__all__ = ["LockLostError", "LockStore", "LockToken"]
