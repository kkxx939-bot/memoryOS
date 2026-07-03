from __future__ import annotations


class MemoryOSError(Exception):
    """Base error for production MemoryOS components."""


class InvalidContextURI(MemoryOSError, ValueError):
    """Raised when a memoryos:// URI is malformed or unsafe."""


class PolicyBlocked(MemoryOSError):
    """Raised when a policy gate blocks execution."""
