"""Explicit boundary for the unavailable filesystem lease-store backend."""

from memoryos.adapters.locks.errors import LockBackendUnavailableError


class FileSystemLockStore:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise LockBackendUnavailableError(
            "FileSystemLockStore is not implemented; use SQLiteLockStore or provide an explicit LockStore"
        )


__all__ = ["FileSystemLockStore"]
