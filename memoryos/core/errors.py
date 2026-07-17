"""核心工具里的异常。"""

from __future__ import annotations


class MemoryOSError(Exception):
    """MemoryOSError 对应的异常。"""


class InvalidContextURI(MemoryOSError, ValueError):
    """负责 InvalidContextURI 这部分逻辑。"""


class PolicyBlocked(MemoryOSError):
    """负责 PolicyBlocked 这部分逻辑。"""


class RevisionConflictError(RuntimeError):
    """A durable commit no longer matches the revision it planned against."""

    def __init__(self, message: str, *, committed_diff=None) -> None:  # noqa: ANN001
        self.committed_diff = committed_diff
        super().__init__(message)
