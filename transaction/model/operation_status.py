"""统一事务内核中的操作状态。"""

from __future__ import annotations

from enum import Enum


class OperationStatus(str, Enum):
    CANDIDATE = "candidate"
    RESOLVED = "resolved"
    PENDING = "pending"
    COMMITTED = "committed"
    REJECTED = "rejected"
    FAILED = "failed"
    NOOP = "noop"
