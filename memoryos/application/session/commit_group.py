"""Greenfield durable commit groups for independent Session consumers."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from memoryos.core.clock import utc_now
from memoryos.core.durable_io import atomic_write_json
from memoryos.core.durable_io.atomic_file import _open_control_parent
from memoryos.core.durable_io.quarantine import quarantine_control_file
from memoryos.core.file_lock import open_private_lock
from memoryos.core.ids import require_safe_path_segment
from memoryos.core.integrity import canonical_digest
from memoryos.core.path_safety import DurablePathIntegrityError
from memoryos.memory.documents.frontmatter import validate_document_id

try:  # pragma: no cover - production platforms provide fcntl.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

CONSUMERS = ("memory", "behavior", "action_policy", "context")
_SCHEMA = "session_commit_group_v1"
_MAX_CONTROL_BYTES = 2 * 1024 * 1024
_TERMINAL = {"completed", "dead_letter", "quarantine"}


class CommitGroupIntegrityError(RuntimeError):
    """A commit-group control record was malformed and quarantined."""


@dataclass(frozen=True)
class MemoryDocumentEffect:
    """The only persisted memory effect: no Markdown or rendered plan bytes."""

    document_id: str
    change_event_id: str
    change_digest: str

    def __post_init__(self) -> None:
        validate_document_id(self.document_id)
        if not self.change_event_id.startswith("memchg_") or not _is_hex(
            self.change_event_id.removeprefix("memchg_"), 64
        ):
            raise ValueError("memory document change event ID is invalid")
        _require_digest(self.change_digest, "document change digest")

    def to_dict(self) -> dict[str, str]:
        return {
            "document_id": self.document_id,
            "change_event_id": self.change_event_id,
            "change_digest": self.change_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MemoryDocumentEffect:
        return cls(
            document_id=str(payload["document_id"]),
            change_event_id=str(payload["change_event_id"]),
            change_digest=str(payload["change_digest"]),
        )


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
    memory_effects: list[MemoryDocumentEffect] = field(default_factory=list)
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
        if len({effect.change_event_id for effect in self.memory_effects}) != len(self.memory_effects):
            raise ValueError("commit group has duplicate memory document change event IDs")
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

    @property
    def memory_committed(self) -> bool:
        return self.consumers["memory"].status == "completed"

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
            "memory_effects": [effect.to_dict() for effect in self.memory_effects],
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
        raw_effects = payload.get("memory_effects")
        if not isinstance(raw_consumers, dict) or not isinstance(raw_effects, list):
            raise ValueError("commit group consumer or memory effect collection is malformed")
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
            memory_effects=[MemoryDocumentEffect.from_dict(_mapping(item, "memory effect")) for item in raw_effects],
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
        )


class CommitGroupStore:
    """Create-bound group identity with independent consumer leases and retries."""

    MAX_ATTEMPTS = 3

    def __init__(self, root: str | Path, test_hook=None) -> None:  # noqa: ANN001
        self.artifact_root = Path(root).expanduser().resolve(strict=False)
        self.root = self.artifact_root / "system" / "commit_groups"
        self._fallback_locks: dict[str, threading.RLock] = {}
        self._fallback_guard = threading.Lock()
        self.test_hook = test_hook

    def path(self, group_id: str) -> Path:
        return self.root / f"{require_safe_path_segment(group_id, 'commit group_id')}.json"

    def load(self, group_id: str) -> CommitGroupStatus | None:
        return self._load_unlocked(group_id)

    def create(
        self,
        group_id: str,
        *,
        task_id: str,
        archive_uri: str,
        user_id: str,
        tenant_id: str,
        archive_digest: str,
        manifest_digest: str,
    ) -> CommitGroupStatus:
        with self.group_lock(group_id):
            existing = self._load_unlocked(group_id)
            if existing is not None:
                identity = (
                    task_id,
                    archive_uri,
                    user_id,
                    tenant_id,
                    archive_digest,
                    manifest_digest,
                )
                persisted = (
                    existing.task_id,
                    existing.archive_uri,
                    existing.user_id,
                    existing.tenant_id,
                    existing.archive_digest,
                    existing.manifest_digest,
                )
                if persisted != identity:
                    raise ValueError("commit group ID is already bound to another immutable archive")
                return existing
            now = utc_now()
            status = CommitGroupStatus(
                group_id=group_id,
                task_id=task_id,
                archive_uri=archive_uri,
                user_id=user_id,
                tenant_id=tenant_id,
                archive_digest=archive_digest,
                manifest_digest=manifest_digest,
                created_at=now,
                updated_at=now,
            )
            self._write(status)
            return status

    def claim_consumer(
        self,
        group_id: str,
        consumer: str,
        *,
        attempt_id: str,
        lease_seconds: int = 300,
    ) -> bool:
        require_safe_path_segment(attempt_id, "commit consumer attempt_id")
        with self.group_lock(group_id):
            status = self._required_unlocked(group_id)
            item = self._consumer(status, consumer)
            if item.status in _TERMINAL or (item.status == "failed" and not item.retryable):
                return False
            if item.attempt_count >= self.MAX_ATTEMPTS:
                item.status = "dead_letter"
                item.retryable = False
                item.terminal_status = "dead_letter"
                status.updated_at = utc_now()
                self._write(status)
                return False
            if self._lease_active(item.next_retry_at):
                return False
            if item.status == "running" and self._lease_active(item.lease_expires_at):
                return False
            item.status = "running"
            item.attempt_count += 1
            item.last_error = ""
            item.attempt_id = attempt_id
            item.owner_pid = os.getpid()
            item.next_retry_at = ""
            item.terminal_status = ""
            item.lease_expires_at = (datetime.now(timezone.utc) + timedelta(seconds=max(1, lease_seconds))).isoformat()
            status.updated_at = utc_now()
            self._write(status)
            return True

    def start_consumer(self, group_id: str, consumer: str) -> CommitGroupStatus:
        self.claim_consumer(group_id, consumer, attempt_id=uuid.uuid4().hex)
        return self._required_unlocked(group_id)

    def record_memory_effect(
        self,
        group_id: str,
        effect: MemoryDocumentEffect,
        *,
        attempt_id: str,
    ) -> CommitGroupStatus:
        with self.group_lock(group_id):
            status = self._required_unlocked(group_id)
            memory = self._consumer(status, "memory")
            self._assert_attempt(memory, attempt_id)
            existing = next(
                (item for item in status.memory_effects if item.change_event_id == effect.change_event_id),
                None,
            )
            if existing is not None:
                if existing != effect:
                    raise CommitGroupIntegrityError(
                        "memory change event ID is bound to another document effect"
                    )
                return status
            status.memory_effects.append(effect)
            status.updated_at = utc_now()
            self._write(status)
            if self.test_hook is not None:
                self.test_hook("after_memory_effect_record", group_id)
            return status

    def complete_consumer(
        self,
        group_id: str,
        consumer: str,
        *,
        attempt_id: str,
        summary: Mapping[str, Any] | None = None,
    ) -> CommitGroupStatus:
        resolved_summary = dict(summary or {})
        _validate_summary(consumer, resolved_summary)
        with self.group_lock(group_id):
            status = self._required_unlocked(group_id)
            item = self._consumer(status, consumer)
            if item.status == "completed":
                if item.summary != resolved_summary:
                    raise CommitGroupIntegrityError("completed consumer summary changed across replay")
                return status
            self._assert_attempt(item, attempt_id)
            item.status = "completed"
            item.retryable = False
            item.last_error = ""
            item.attempt_id = ""
            item.owner_pid = 0
            item.lease_expires_at = ""
            item.next_retry_at = ""
            item.terminal_status = "done"
            item.summary = resolved_summary
            status.updated_at = utc_now()
            self._write(status)
            return status

    def fail_consumer(
        self,
        group_id: str,
        consumer: str,
        error: str,
        *,
        retryable: bool,
        attempt_id: str,
    ) -> CommitGroupStatus:
        with self.group_lock(group_id):
            status = self._required_unlocked(group_id)
            item = self._consumer(status, consumer)
            if item.status == "completed":
                return status
            self._assert_attempt(item, attempt_id)
            exhausted = item.attempt_count >= self.MAX_ATTEMPTS
            item.status = "failed" if retryable and not exhausted else "dead_letter"
            item.retryable = retryable and not exhausted
            item.last_error = _content_free_error(error)
            item.attempt_id = ""
            item.owner_pid = 0
            item.lease_expires_at = ""
            item.next_retry_at = utc_now() if item.retryable else ""
            item.terminal_status = "" if item.retryable else "dead_letter"
            status.updated_at = utc_now()
            self._write(status)
            return status

    def pending(self) -> list[CommitGroupStatus]:
        return [status for status in self.all() if not status.terminal]

    def all(self) -> list[CommitGroupStatus]:
        result: list[CommitGroupStatus] = []
        descriptor = _open_control_parent(self.root / ".scan", self.artifact_root)
        try:
            names = sorted(name for name in os.listdir(descriptor) if name.endswith(".json"))
        finally:
            os.close(descriptor)
        for name in names:
            group_id = name.removesuffix(".json")
            require_safe_path_segment(group_id, "commit group_id")
            status = self._load_unlocked(group_id)
            if status is None:
                raise CommitGroupIntegrityError(f"commit group disappeared during scan: {name}")
            result.append(status)
        return result

    def recover_expired_consumers(self) -> list[tuple[str, str]]:
        recovered: list[tuple[str, str]] = []
        for snapshot in self.pending():
            with self.group_lock(snapshot.group_id):
                status = self._required_unlocked(snapshot.group_id)
                changed = False
                for name, item in status.consumers.items():
                    if item.status != "running" or self._lease_active(item.lease_expires_at):
                        continue
                    self._release_consumer(item, "consumer_lease_expired")
                    recovered.append((status.group_id, name))
                    changed = True
                if changed:
                    status.updated_at = utc_now()
                    self._write(status)
        return recovered

    def recover_abandoned_leases(self) -> list[tuple[str, str]]:
        recovered: list[tuple[str, str]] = []
        for snapshot in self.pending():
            with self.group_lock(snapshot.group_id):
                status = self._required_unlocked(snapshot.group_id)
                changed = False
                for name, item in status.consumers.items():
                    if item.status != "running" or item.owner_pid <= 0 or self._pid_alive(item.owner_pid):
                        continue
                    self._release_consumer(item, "consumer_owner_exited")
                    recovered.append((status.group_id, name))
                    changed = True
                if changed:
                    status.updated_at = utc_now()
                    self._write(status)
        return recovered

    @contextmanager
    def group_lock(self, group_id: str) -> Iterator[None]:
        safe_group = require_safe_path_segment(group_id, "commit group_id")
        lock_path = self.root / ".locks" / f"{safe_group}.lock"
        if fcntl is not None:
            descriptor = open_private_lock(lock_path, root=self.root)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
            return
        with self._fallback_guard:  # pragma: no cover
            lock = self._fallback_locks.setdefault(str(lock_path), threading.RLock())
        with lock:  # pragma: no cover
            yield

    def _write(self, status: CommitGroupStatus) -> None:
        payload = status.to_dict()
        payload["control_digest"] = canonical_digest(payload)
        atomic_write_json(self.path(status.group_id), payload, artifact_root=self.artifact_root)

    def _load_unlocked(self, group_id: str) -> CommitGroupStatus | None:
        path = self.path(group_id)
        try:
            payload = self._read_payload(path, self.artifact_root)
            digest = payload.get("control_digest")
            core = {key: value for key, value in payload.items() if key != "control_digest"}
            if digest != canonical_digest(core) or payload.get("group_id") != group_id:
                raise ValueError("commit group digest or path identity is invalid")
            return CommitGroupStatus.from_dict(payload)
        except FileNotFoundError:
            return None
        except (
            OSError,
            DurablePathIntegrityError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            try:
                quarantine_control_file(
                    self.artifact_root,
                    path,
                    kind="session_commit_group",
                    error=exc,
                    identifiers={"group_id": group_id},
                )
            except (OSError, DurablePathIntegrityError):
                # An unsafe parent may make quarantine publication impossible;
                # preserve the original fail-closed integrity classification.
                pass
            raise CommitGroupIntegrityError("commit group state quarantined") from exc

    @staticmethod
    def _read_payload(path: Path, artifact_root: Path) -> dict[str, Any]:
        parent_descriptor = _open_control_parent(path, artifact_root)
        try:
            descriptor = os.open(
                path.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise ValueError("commit group state is not one regular file")
                if metadata.st_size > _MAX_CONTROL_BYTES:
                    raise ValueError("commit group state exceeds its size bound")
                chunks: list[bytes] = []
                remaining = _MAX_CONTROL_BYTES + 1
                while remaining:
                    chunk = os.read(descriptor, min(65536, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
            finally:
                os.close(descriptor)
        finally:
            os.close(parent_descriptor)
        raw = b"".join(chunks)
        if len(raw) > _MAX_CONTROL_BYTES:
            raise ValueError("commit group state exceeds its size bound")
        payload = json.loads(raw.decode("utf-8", errors="strict"))
        return _mapping(payload, "commit group")

    def _required_unlocked(self, group_id: str) -> CommitGroupStatus:
        status = self._load_unlocked(group_id)
        if status is None:
            raise KeyError(f"unknown commit group: {group_id}")
        return status

    @staticmethod
    def _consumer(status: CommitGroupStatus, consumer: str) -> ConsumerStatus:
        if consumer not in CONSUMERS:
            raise ValueError(f"unsupported commit-group consumer: {consumer}")
        return status.consumers[consumer]

    @staticmethod
    def _assert_attempt(item: ConsumerStatus, attempt_id: str) -> None:
        if item.status != "running" or not attempt_id or item.attempt_id != attempt_id:
            raise RuntimeError("commit-group consumer attempt no longer owns its lease")

    @staticmethod
    def _lease_active(value: str) -> bool:
        if not value:
            return False
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return False
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc) > datetime.now(timezone.utc)

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @staticmethod
    def _release_consumer(item: ConsumerStatus, error: str) -> None:
        item.status = "failed"
        item.retryable = True
        item.last_error = error
        item.attempt_id = ""
        item.owner_pid = 0
        item.lease_expires_at = ""
        item.next_retry_at = utc_now()


def _validate_summary(consumer: str, summary: Mapping[str, Any]) -> None:
    allowed = {
        "memory": {
            "status",
            "edit_proposal_count",
            "edit_proposal_ids",
            "document_change_count",
            "no_op_count",
        },
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
        if key in {"edit_proposal_count", "document_change_count", "no_op_count", "operation_count"} and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError("commit-group summary counter is invalid")
        if key == "operation_ids" and (
            not isinstance(value, list) or any(not isinstance(item, str) or not _is_identifier(item) for item in value)
        ):
            raise ValueError("commit-group operation IDs are invalid")
        if key == "edit_proposal_ids" and (
            not isinstance(value, list)
            or any(not isinstance(item, str) or not item.startswith("mdreview_") or not _is_identifier(item) for item in value)
        ):
            raise ValueError("commit-group edit proposal IDs are invalid")
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
    "CommitGroupIntegrityError",
    "CommitGroupStatus",
    "CommitGroupStore",
    "ConsumerStatus",
    "MemoryDocumentEffect",
]
