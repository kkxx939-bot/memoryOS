"""Session 派生消费者的耐久提交组存储。"""

from __future__ import annotations

import json
import os
import stat
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from foundation.clock import utc_now
from foundation.ids import require_safe_path_segment
from foundation.integrity import canonical_digest
from infrastructure.store.filesystem.durable_io import atomic_write_json
from infrastructure.store.filesystem.durable_io.atomic_file import _open_control_parent
from infrastructure.store.filesystem.durable_io.quarantine import quarantine_control_file
from infrastructure.store.filesystem.file_lock import open_private_lock
from infrastructure.store.filesystem.path_safety import DurablePathIntegrityError
from infrastructure.store.session.commit_group_model import (
    _TERMINAL,
    CONSUMERS,
    CommitGroupStatus,
    ConsumerStatus,
    _content_free_error,
    _mapping,
    _validate_summary,
)

try:  # pragma: no cover - 生产平台提供 fcntl
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

_MAX_CONTROL_BYTES = 2 * 1024 * 1024


class CommitGroupIntegrityError(RuntimeError):
    """提交组控制记录格式损坏并已隔离。"""




class CommitGroupStore:
    """保存创建即绑定的提交组身份，以及独立消费者租约和重试状态。"""

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
                # 不安全的父目录可能导致隔离记录无法发布；此时保留原始失败关闭分类。
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



__all__ = [
    "CONSUMERS",
    "CommitGroupIntegrityError",
    "CommitGroupStatus",
    "CommitGroupStore",
    "ConsumerStatus",
]
