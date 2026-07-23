"""Session 提交组的纯数据模型与内容安全校验。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from foundation.ids import require_safe_path_segment

CONSUMERS = ("behavior", "action_policy", "context")
_SCHEMA = "session_commit_group_v2"
_TERMINAL = {"completed", "dead_letter", "quarantine"}


@dataclass
class ConsumerStatus:
    status: str = "pending"
    attempt_count: int = 0
    last_error: str = ""
    retryable: bool = True
    attempt_id: str = ""
    owner_pid: int = 0
    lease_expires_at: str = ""
    next_retry_at: str = ""
    terminal_status: str = ""
    summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"pending", "running", "failed", *_TERMINAL}:
            raise ValueError("commit-group consumer status is invalid")
        if self.attempt_count < 0 or self.owner_pid < 0:
            raise ValueError("commit-group consumer counters are invalid")
        if self.last_error and self.last_error != _content_free_error(self.last_error):
            raise ValueError("commit-group consumer error must be a content-free code")
        if self.attempt_id and not _is_identifier(self.attempt_id):
            raise ValueError("commit-group consumer attempt ID is invalid")
        if self.terminal_status not in {"", "done", "dead_letter", "quarantine"}:
            raise ValueError("commit-group consumer terminal status is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "attempt_count": self.attempt_count,
            "last_error": self.last_error,
            "retryable": self.retryable,
            "attempt_id": self.attempt_id,
            "owner_pid": self.owner_pid,
            "lease_expires_at": self.lease_expires_at,
            "next_retry_at": self.next_retry_at,
            "terminal_status": self.terminal_status,
            "summary": dict(self.summary),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConsumerStatus:
        summary = payload.get("summary", {})
        if not isinstance(summary, dict):
            raise ValueError("commit-group consumer summary must be an object")
        return cls(
            status=str(payload.get("status") or "pending"),
            attempt_count=int(payload.get("attempt_count", 0)),
            last_error=str(payload.get("last_error") or ""),
            retryable=bool(payload.get("retryable", True)),
            attempt_id=str(payload.get("attempt_id") or ""),
            owner_pid=int(payload.get("owner_pid", 0)),
            lease_expires_at=str(payload.get("lease_expires_at") or ""),
            next_retry_at=str(payload.get("next_retry_at") or ""),
            terminal_status=str(payload.get("terminal_status") or ""),
            summary=dict(summary),
        )


@dataclass
class CommitGroupStatus:
    group_id: str
    task_id: str
    archive_uri: str
    user_id: str
    tenant_id: str
    archive_digest: str
    manifest_digest: str
    consumers: dict[str, ConsumerStatus] = field(default_factory=lambda: {name: ConsumerStatus() for name in CONSUMERS})
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        require_safe_path_segment(self.group_id, "commit group_id")
        require_safe_path_segment(self.task_id, "commit task_id")
        require_safe_path_segment(self.user_id, "commit user_id")
        require_safe_path_segment(self.tenant_id, "commit tenant_id")
        _require_digest(self.archive_digest, "archive digest")
        _require_digest(self.manifest_digest, "archive manifest digest")
        if not self.archive_uri.startswith(f"memoryos://user/{self.user_id}/sessions/"):
            raise ValueError("commit group archive URI crosses its owner boundary")
        if set(self.consumers) != set(CONSUMERS):
            raise ValueError("commit group consumers do not match the greenfield schema")
        for name, consumer in self.consumers.items():
            _validate_summary(name, consumer.summary)
        if not self.created_at or not self.updated_at:
            raise ValueError("commit group timestamps must be non-empty")

    @property
    def complete(self) -> bool:
        return all(item.status == "completed" for item in self.consumers.values())

    @property
    def terminal(self) -> bool:
        return all(item.status in _TERMINAL for item in self.consumers.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "group_id": self.group_id,
            "task_id": self.task_id,
            "archive_uri": self.archive_uri,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "archive_digest": self.archive_digest,
            "manifest_digest": self.manifest_digest,
            "consumers": {name: self.consumers[name].to_dict() for name in CONSUMERS},
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "complete": self.complete,
            "terminal": self.terminal,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CommitGroupStatus:
        if payload.get("schema") != _SCHEMA:
            raise ValueError("commit group schema is unsupported; reset is required")
        raw_consumers = payload.get("consumers")
        if not isinstance(raw_consumers, dict):
            raise ValueError("commit group consumer collection is malformed")
        consumers = {
            name: ConsumerStatus.from_dict(_mapping(raw_consumers.get(name), f"{name} consumer")) for name in CONSUMERS
        }
        return cls(
            group_id=str(payload["group_id"]),
            task_id=str(payload["task_id"]),
            archive_uri=str(payload["archive_uri"]),
            user_id=str(payload["user_id"]),
            tenant_id=str(payload["tenant_id"]),
            archive_digest=str(payload["archive_digest"]),
            manifest_digest=str(payload["manifest_digest"]),
            consumers=consumers,
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
        )

def _validate_summary(consumer: str, summary: Mapping[str, Any]) -> None:
    allowed = {
        "behavior": {"status", "operation_count", "operation_ids", "diff_id", "skipped"},
        "action_policy": {"status", "operation_count", "operation_ids", "diff_id", "skipped"},
        "context": {"status", "operation_count", "operation_ids", "diff_id", "skipped"},
    }
    if consumer not in allowed or set(summary) - allowed[consumer]:
        raise ValueError("commit-group consumer summary contains unsupported or content-bearing fields")
    for key, value in summary.items():
        if key in {"status", "diff_id"} and not isinstance(value, str):
            raise ValueError("commit-group summary string field is invalid")
        if key == "status" and (not value or value != _content_free_error(value)):
            raise ValueError("commit-group summary status must be a content-free code")
        if key == "diff_id" and value and not _is_identifier(value):
            raise ValueError("commit-group diff ID is invalid")
        if key == "operation_count" and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError("commit-group summary counter is invalid")
        if key == "operation_ids" and (
            not isinstance(value, list) or any(not isinstance(item, str) or not _is_identifier(item) for item in value)
        ):
            raise ValueError("commit-group operation IDs are invalid")
        if key == "skipped" and not isinstance(value, bool):
            raise ValueError("commit-group skipped flag is invalid")


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_digest(value: str, label: str) -> None:
    if not _is_hex(value, 64):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _is_hex(value: str, length: int) -> bool:
    return len(value) == length and all(character in "0123456789abcdef" for character in value)


def _content_free_error(value: object) -> str:
    text = str(value or "")
    if 0 < len(text) <= 120 and all(character.isalnum() or character in "_.:-" for character in text):
        return text
    return f"error_{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _is_identifier(value: str) -> bool:
    return 0 < len(value) <= 256 and all(character.isalnum() or character in "._:-" for character in value)



__all__ = [
    "CONSUMERS",
    "CommitGroupStatus",
    "ConsumerStatus",
]
