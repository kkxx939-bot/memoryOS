"""核心工具里的异常。"""

from __future__ import annotations


class MemoryOSError(Exception):
    """MemoryOSError 对应的异常。"""


class InvalidContextURI(MemoryOSError, ValueError):
    """负责 InvalidContextURI 这部分逻辑。"""


class PolicyBlocked(MemoryOSError):
    """负责 PolicyBlocked 这部分逻辑。"""
