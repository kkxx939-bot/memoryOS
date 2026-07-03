from __future__ import annotations

from enum import Enum


class LifecycleState(str, Enum):
    CANDIDATE = "candidate"
    RESOLVED = "resolved"
    PENDING = "pending"
    COMMITTED = "committed"
    ACTIVE = "active"
    COLD = "cold"
    ARCHIVED = "archived"
    OBSOLETE = "obsolete"
    DELETED = "deleted"
    REJECTED = "rejected"
