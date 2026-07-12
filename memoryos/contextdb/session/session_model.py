"""上下文数据库里的会话数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from memoryos.core.ids import new_id
from memoryos.core.time import utc_now


@dataclass
class SessionArchive:
    user_id: str
    session_id: str
    archive_uri: str
    messages: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)
    predictions: list[dict] = field(default_factory=list)
    action_results: list[dict] = field(default_factory=list)
    feedback: list[dict] = field(default_factory=list)
    used_contexts: list[dict] = field(default_factory=list)
    used_skills: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    task_id: str = field(default_factory=lambda: new_id("session_commit"))
    created_at: str = field(default_factory=utc_now)
    schema_version: str = "session_archive_v2"
    archive_digest: str = ""
    manifest_digest: str = ""
    manifest_uri: str = ""

    def manifest(self) -> dict:
        return {
            "task_id": self.task_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "archive_uri": self.archive_uri,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
            "archive_digest": self.archive_digest,
            "manifest_digest": self.manifest_digest,
            "manifest_uri": self.manifest_uri,
            "metadata": self.metadata,
            "phase": "sync_archive",
            "files": [
                "commit_head.json",
                "evidence/events/",
                "evidence/objects/",
                "evidence/manifests/",
            ],
        }


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
    canonical_committed: bool = False
    commit_group_status: dict[str, Any] = field(default_factory=dict)
    archive_committed: bool = False
    canonical_active_operation_count: int = 0
    pending_count: int = 0
    pending_persisted: bool = False
