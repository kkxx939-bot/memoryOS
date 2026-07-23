"""Session 提交状态与提交结果模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SessionCommitState(str, Enum):
    OPEN = "OPEN"
    ARCHIVED = "ARCHIVED"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMMITTED = "COMMITTED"
    FAILED_RETRYABLE = "FAILED_RETRYABLE"
    DEAD_LETTER = "DEAD_LETTER"


@dataclass(frozen=True)
class SessionCommitResult:
    task_id: str
    archive_uri: str
    status: str
    done: bool = False
    state: SessionCommitState = SessionCommitState.QUEUED
    commit_group_id: str = ""
    commit_group_status: dict[str, Any] = field(default_factory=dict)
    archive_committed: bool = False
    session_projection_status: str = "not_configured"
    session_projected_count: int = 0
