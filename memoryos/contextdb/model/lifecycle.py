"""上下文数据库里的生命周期。"""

from __future__ import annotations

from enum import Enum


class LifecycleState(str, Enum):
    CANDIDATE = "candidate"
    RESOLVED = "resolved"
    PENDING = "pending"
    CONFIRMED = "confirmed"
    RETRYABLE = "retryable"
    COMMITTED = "committed"
    ACTIVE = "active"
    COLD = "cold"
    ARCHIVED = "archived"
    OBSOLETE = "obsolete"
    DELETED = "deleted"
    REJECTED = "rejected"
    EXPIRED = "expired"
