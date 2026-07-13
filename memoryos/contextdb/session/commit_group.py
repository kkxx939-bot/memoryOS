"""Durable status for the post-canonical session commit group."""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from memoryos.core.file_lock import open_private_lock
from memoryos.core.ids import require_safe_path_segment
from memoryos.core.time import utc_now
from memoryos.memory.canonical.event import canonical_digest
from memoryos.operations.commit.effect_marker import atomic_write_json
from memoryos.operations.commit.quarantine import quarantine_control_file

try:  # pragma: no cover - supported production platforms provide fcntl.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


CONSUMERS = ("projection", "behavior", "action_policy", "context")


class CommitGroupIntegrityError(RuntimeError):
    """A corrupt commit-group control file was quarantined."""


@dataclass
class ConsumerStatus:
    status: str = "pending"
    attempt_count: int = 0
    last_error: str = ""
    retryable: bool = True
    completed_revision: int | None = None
    attempt_id: str = ""
    owner_pid: int = 0
    lease_expires_at: str = ""
    next_retry_at: str = ""
    terminal_status: str = ""
    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "attempt_count": self.attempt_count,
            "last_error": self.last_error,
            "retryable": self.retryable,
            "completed_revision": self.completed_revision,
            "attempt_id": self.attempt_id,
            "owner_pid": self.owner_pid,
            "lease_expires_at": self.lease_expires_at,
            "next_retry_at": self.next_retry_at,
            "terminal_status": self.terminal_status,
            "result": dict(self.result),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConsumerStatus:
        return cls(
            status=str(payload.get("status", "pending")),
            attempt_count=int(payload.get("attempt_count", 0)),
            last_error=str(payload.get("last_error", "")),
            retryable=bool(payload.get("retryable", True)),
            completed_revision=(
                int(payload["completed_revision"]) if payload.get("completed_revision") is not None else None
            ),
            attempt_id=str(payload.get("attempt_id", "")),
            owner_pid=int(payload.get("owner_pid", 0) or 0),
            lease_expires_at=str(payload.get("lease_expires_at", "")),
            next_retry_at=str(payload.get("next_retry_at", "")),
            terminal_status=str(payload.get("terminal_status", "")),
            result=dict(payload.get("result", {}) or {}),
        )


@dataclass
class CommitGroupStatus:
    group_id: str
    task_id: str
    archive_uri: str
    user_id: str
    tenant_id: str
    archive_digest: str = ""
    manifest_digest: str = ""
    canonical_status: str = "pending"
    canonical_revision: int | None = None
    canonical_attempt_count: int = 0
    canonical_last_error: str = ""
    canonical_retryable: bool = True
    canonical_result: dict[str, Any] = field(default_factory=dict)
    canonical_effects: dict[str, dict[str, Any]] = field(default_factory=dict)
    canonical_attempt_id: str = ""
    canonical_owner_pid: int = 0
    canonical_lease_expires_at: str = ""
    canonical_next_retry_at: str = ""
    canonical_terminal_status: str = ""
    canonical_phase: str = "unstarted"
    salience_reservation_digest: str = ""
    planning_digest: str = ""
    consumers: dict[str, ConsumerStatus] = field(default_factory=lambda: {name: ConsumerStatus() for name in CONSUMERS})
    created_at: str = ""
    updated_at: str = ""

    @property
    def complete(self) -> bool:
        return self.canonical_status == "completed" and all(
            item.status == "completed" for item in self.consumers.values()
        )

    @property
    def terminal(self) -> bool:
        if self.canonical_status in {"dead_letter", "quarantine"}:
            return True
        return self.canonical_status == "completed" and all(
            item.status in {"completed", "dead_letter", "quarantine"}
            for item in self.consumers.values()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "task_id": self.task_id,
            "archive_uri": self.archive_uri,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "archive_digest": self.archive_digest,
            "manifest_digest": self.manifest_digest,
            "canonical_status": self.canonical_status,
            "canonical_revision": self.canonical_revision,
            "canonical_attempt_count": self.canonical_attempt_count,
            "canonical_last_error": self.canonical_last_error,
            "canonical_retryable": self.canonical_retryable,
            "canonical_result": dict(self.canonical_result),
            "canonical_effects": {
                key: dict(value) for key, value in self.canonical_effects.items()
            },
            "canonical_attempt_id": self.canonical_attempt_id,
            "canonical_owner_pid": self.canonical_owner_pid,
            "canonical_lease_expires_at": self.canonical_lease_expires_at,
            "canonical_next_retry_at": self.canonical_next_retry_at,
            "canonical_terminal_status": self.canonical_terminal_status,
            "canonical_phase": self.canonical_phase,
            "salience_reservation_digest": self.salience_reservation_digest,
            "planning_digest": self.planning_digest,
            "consumers": {key: value.to_dict() for key, value in self.consumers.items()},
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "complete": self.complete,
            "terminal": self.terminal,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CommitGroupStatus:
        statuses = {
            name: ConsumerStatus.from_dict(dict(payload.get("consumers", {}).get(name, {}) or {})) for name in CONSUMERS
        }
        return cls(
            group_id=str(payload["group_id"]),
            task_id=str(payload["task_id"]),
            archive_uri=str(payload["archive_uri"]),
            user_id=str(payload["user_id"]),
            tenant_id=str(payload.get("tenant_id", "default")),
            archive_digest=str(payload.get("archive_digest", "")),
            manifest_digest=str(payload.get("manifest_digest", "")),
            canonical_status=str(payload.get("canonical_status", "pending")),
            canonical_revision=(
                int(payload["canonical_revision"]) if payload.get("canonical_revision") is not None else None
            ),
            canonical_attempt_count=int(payload.get("canonical_attempt_count", 0)),
            canonical_last_error=str(payload.get("canonical_last_error", "")),
            canonical_retryable=bool(payload.get("canonical_retryable", True)),
            canonical_result=dict(payload.get("canonical_result", {}) or {}),
            canonical_effects={
                str(key): dict(value)
                for key, value in dict(payload.get("canonical_effects", {}) or {}).items()
                if isinstance(value, dict)
            },
            canonical_attempt_id=str(payload.get("canonical_attempt_id", "")),
            canonical_owner_pid=int(payload.get("canonical_owner_pid", 0) or 0),
            canonical_lease_expires_at=str(payload.get("canonical_lease_expires_at", "")),
            canonical_next_retry_at=str(payload.get("canonical_next_retry_at", "")),
            canonical_terminal_status=str(payload.get("canonical_terminal_status", "")),
            canonical_phase=str(payload.get("canonical_phase", "unstarted")),
            salience_reservation_digest=str(payload.get("salience_reservation_digest", "")),
            planning_digest=str(payload.get("planning_digest", "")),
            consumers=statuses,
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
        )


class CommitGroupStore:
    """Create-only group identity with atomic, idempotent status updates."""

    MAX_ATTEMPTS = 3

    def __init__(self, root: str | Path, test_hook=None) -> None:  # noqa: ANN001
        self.artifact_root = Path(root)
        self.root = self.artifact_root / "system" / "commit_groups"
        self._fallback_locks: dict[str, threading.RLock] = {}
        self._fallback_guard = threading.Lock()
        self.test_hook = test_hook

    def path(self, group_id: str) -> Path:
        safe_group = require_safe_path_segment(group_id, "commit group_id")
        return self.root / f"{safe_group}.json"

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
        archive_digest: str = "",
        manifest_digest: str = "",
    ) -> CommitGroupStatus:
        require_safe_path_segment(group_id, "commit group_id")
        require_safe_path_segment(task_id, "commit task_id")
        require_safe_path_segment(user_id, "commit user_id")
        require_safe_path_segment(tenant_id, "commit tenant_id")
        with self.group_lock(group_id):
            existing = self._load_unlocked(group_id)
            if existing is not None:
                if (
                    existing.task_id != task_id
                    or existing.archive_uri != archive_uri
                    or existing.user_id != user_id
                    or existing.tenant_id != tenant_id
                    or (existing.archive_digest and archive_digest and existing.archive_digest != archive_digest)
                    or (existing.manifest_digest and manifest_digest and existing.manifest_digest != manifest_digest)
                ):
                    raise ValueError("commit group id is already bound to another request")
                return existing
            now = _now()
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

    def start_canonical(self, group_id: str) -> CommitGroupStatus:
        self.claim_canonical(group_id, attempt_id=uuid.uuid4().hex)
        return self._required(group_id)

    def claim_canonical(
        self,
        group_id: str,
        *,
        attempt_id: str,
        lease_seconds: int = 300,
    ) -> bool:
        with self.group_lock(group_id):
            status = self._required_unlocked(group_id)
            if status.canonical_status == "completed" or (
                status.canonical_status in {"dead_letter", "quarantine"}
                or (status.canonical_status == "failed" and not status.canonical_retryable)
            ):
                return False
            if status.canonical_attempt_count >= self.MAX_ATTEMPTS:
                status.canonical_status = "dead_letter"
                status.canonical_retryable = False
                status.canonical_terminal_status = "dead_letter"
                status.updated_at = _now()
                self._write(status)
                return False
            if self._retry_waiting(status.canonical_next_retry_at):
                return False
            if status.canonical_status == "running" and self._lease_active(status.canonical_lease_expires_at):
                return False
            status.canonical_status = "running"
            status.canonical_attempt_count += 1
            status.canonical_last_error = ""
            status.canonical_attempt_id = attempt_id
            status.canonical_owner_pid = os.getpid()
            status.canonical_next_retry_at = ""
            status.canonical_terminal_status = ""
            status.canonical_phase = "claimed"
            status.canonical_lease_expires_at = (
                datetime.now(timezone.utc) + timedelta(seconds=max(1, lease_seconds))
            ).isoformat()
            status.updated_at = _now()
            self._write(status)
            return True

    def mark_canonical_phase(
        self,
        group_id: str,
        *,
        phase: str,
        attempt_id: str,
        salience_reservation_digest: str = "",
        planning_digest: str = "",
    ) -> CommitGroupStatus:
        """Persist the semantic-planning publication boundary under lease fencing."""

        allowed = {"salience_reserved", "planning_sealed"}
        if phase not in allowed:
            raise ValueError(f"unsupported canonical planning phase: {phase}")
        with self.group_lock(group_id):
            status = self._required_unlocked(group_id)
            if status.canonical_status != "running" or status.canonical_attempt_id != attempt_id:
                raise RuntimeError("canonical planning attempt no longer owns the lease")
            if phase == "salience_reserved":
                if len(salience_reservation_digest) != 64:
                    raise ValueError("salience reservation phase requires a durable digest")
                if (
                    status.salience_reservation_digest
                    and status.salience_reservation_digest != salience_reservation_digest
                ):
                    raise ValueError("commit group is bound to another salience reservation")
                status.salience_reservation_digest = salience_reservation_digest
            else:
                if len(salience_reservation_digest) != 64 or len(planning_digest) != 64:
                    raise ValueError("sealed planning phase requires salience and planning digests")
                if (
                    status.salience_reservation_digest
                    and status.salience_reservation_digest != salience_reservation_digest
                ):
                    raise ValueError("sealed planning differs from the reserved salience decision")
                if status.planning_digest and status.planning_digest != planning_digest:
                    raise ValueError("commit group is bound to another planning envelope")
                status.salience_reservation_digest = salience_reservation_digest
                status.planning_digest = planning_digest
            order = {"unstarted": 0, "claimed": 1, "salience_reserved": 2, "planning_sealed": 3}
            if order.get(phase, -1) < order.get(status.canonical_phase, -1):
                raise ValueError("canonical planning phase cannot move backwards")
            status.canonical_phase = phase
            status.updated_at = _now()
            self._write(status)
            return status

    def mark_canonical(
        self,
        group_id: str,
        *,
        revision: int | None = None,
        result: dict[str, Any] | None = None,
        attempt_id: str | None = None,
    ) -> CommitGroupStatus:
        with self.group_lock(group_id):
            status = self._required_unlocked(group_id)
            if attempt_id is not None and status.canonical_attempt_id != attempt_id:
                raise RuntimeError("canonical commit attempt no longer owns the lease")
            status.canonical_status = "completed"
            status.canonical_revision = revision
            status.canonical_last_error = ""
            status.canonical_retryable = False
            status.canonical_result = dict(result or {})
            status.canonical_attempt_id = ""
            status.canonical_owner_pid = 0
            status.canonical_lease_expires_at = ""
            status.canonical_next_retry_at = ""
            status.canonical_terminal_status = "done"
            status.canonical_phase = "committed"
            status.updated_at = _now()
            self._write(status)
            return status

    def fail_canonical(
        self,
        group_id: str,
        error: str,
        *,
        retryable: bool,
        attempt_id: str | None = None,
    ) -> CommitGroupStatus:
        with self.group_lock(group_id):
            status = self._required_unlocked(group_id)
            if attempt_id is not None and status.canonical_attempt_id != attempt_id:
                raise RuntimeError("canonical commit attempt no longer owns the lease")
            if status.canonical_status == "completed":
                return status
            exhausted = status.canonical_attempt_count >= self.MAX_ATTEMPTS
            status.canonical_status = "failed" if retryable and not exhausted else "dead_letter"
            status.canonical_last_error = str(error)[:200]
            status.canonical_retryable = retryable and not exhausted
            status.canonical_attempt_id = ""
            status.canonical_owner_pid = 0
            status.canonical_lease_expires_at = ""
            status.canonical_next_retry_at = _now() if status.canonical_retryable else ""
            status.canonical_terminal_status = "" if status.canonical_retryable else "dead_letter"
            if status.canonical_status == "dead_letter":
                for item in status.consumers.values():
                    if item.status not in {"completed", "quarantine"}:
                        item.status = "dead_letter"
                        item.retryable = False
                        item.terminal_status = "dead_letter"
            status.updated_at = _now()
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
        with self.group_lock(group_id):
            status = self._required_unlocked(group_id)
            item = self._consumer(status, consumer)
            if item.status in {"completed", "dead_letter", "quarantine"} or (
                item.status == "failed" and not item.retryable
            ):
                return False
            if item.attempt_count >= self.MAX_ATTEMPTS:
                item.status = "dead_letter"
                item.retryable = False
                item.terminal_status = "dead_letter"
                status.updated_at = _now()
                self._write(status)
                return False
            if self._retry_waiting(item.next_retry_at):
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
            status.updated_at = _now()
            self._write(status)
            return True

    def start_consumer(self, group_id: str, consumer: str) -> CommitGroupStatus:
        attempt_id = uuid.uuid4().hex
        self.claim_consumer(group_id, consumer, attempt_id=attempt_id)
        status = self._required(group_id)
        return status

    def record_canonical_effect(
        self,
        group_id: str,
        diff: dict[str, Any],
    ) -> CommitGroupStatus:
        """Append one marker-backed canonical diff without changing attempt ownership."""

        diff_id = str(diff.get("diff_id") or "")
        if not diff_id:
            raise ValueError("canonical effect requires a diff_id")
        with self.group_lock(group_id):
            status = self._required_unlocked(group_id)
            if str(diff.get("user_id") or "") != status.user_id:
                raise ValueError("canonical effect user does not match its commit group")
            operations = diff.get("operations", []) or []
            if not isinstance(operations, list) or not operations:
                raise ValueError("canonical effect requires committed operations")
            for operation in operations:
                if not isinstance(operation, dict):
                    raise ValueError("canonical effect operation must be an object")
                payload = operation.get("payload", {})
                if not isinstance(payload, dict):
                    raise ValueError("canonical effect payload must be an object")
                if (
                    str(operation.get("user_id") or "") != status.user_id
                    or not (
                        payload.get("canonical_memory") is True
                        or (
                            payload.get("canonical_pending_proposal") is True
                            and not payload.get("commit_consumer")
                        )
                    )
                    or str(payload.get("commit_group_id") or "") != group_id
                    or str(payload.get("tenant_id") or "default") != status.tenant_id
                ):
                    raise ValueError("canonical effect crosses its commit-group boundary")
            existing = status.canonical_effects.get(diff_id)
            if existing is not None:
                if self._canonical_json(existing) != self._canonical_json(diff):
                    raise ValueError("canonical effect diff id conflicts with another effect")
                return status
            status.canonical_effects[diff_id] = dict(diff)
            status.updated_at = _now()
            self._write(status)
            if self.test_hook is not None:
                self.test_hook("after_commit_group_effect_record", group_id)
            return status

    def complete_consumer(
        self,
        group_id: str,
        consumer: str,
        *,
        revision: int | None = None,
        attempt_id: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> CommitGroupStatus:
        with self.group_lock(group_id):
            status = self._required_unlocked(group_id)
            item = self._consumer(status, consumer)
            self._assert_attempt(item, attempt_id)
            item.status = "completed"
            item.retryable = False
            item.last_error = ""
            item.completed_revision = revision
            item.attempt_id = ""
            item.owner_pid = 0
            item.lease_expires_at = ""
            item.next_retry_at = ""
            item.terminal_status = "done"
            item.result = dict(result or {})
            status.updated_at = _now()
            self._write(status)
            return status

    def refresh_completed_consumer_result(
        self,
        group_id: str,
        consumer: str,
        *,
        result: dict[str, Any],
    ) -> CommitGroupStatus:
        """Refresh a disposable proof only after independently re-verifying it."""

        with self.group_lock(group_id):
            status = self._required_unlocked(group_id)
            item = self._consumer(status, consumer)
            if item.status != "completed" or item.attempt_id or item.owner_pid or item.lease_expires_at:
                raise CommitGroupIntegrityError(
                    "only an unleased completed consumer may receive a refreshed proof"
                )
            item.result = dict(result)
            status.updated_at = _now()
            self._write(status)
            return status

    def fail_consumer(
        self,
        group_id: str,
        consumer: str,
        error: str,
        *,
        retryable: bool = True,
        attempt_id: str | None = None,
    ) -> CommitGroupStatus:
        with self.group_lock(group_id):
            status = self._required_unlocked(group_id)
            item = self._consumer(status, consumer)
            self._assert_attempt(item, attempt_id)
            exhausted = item.attempt_count >= self.MAX_ATTEMPTS
            item.status = "failed" if retryable and not exhausted else "dead_letter"
            item.retryable = retryable and not exhausted
            item.last_error = str(error)[:200]
            item.attempt_id = ""
            item.owner_pid = 0
            item.lease_expires_at = ""
            item.next_retry_at = _now() if item.retryable else ""
            item.terminal_status = "" if item.retryable else "dead_letter"
            status.updated_at = _now()
            self._write(status)
            return status

    def pending(self) -> list[CommitGroupStatus]:
        if not self.root.exists():
            return []
        result = []
        for path in sorted(self.root.glob("*.json")):
            status = self._load_unlocked(path.stem)
            if status is None:
                continue
            if not status.terminal:
                result.append(status)
        return result

    def all(self) -> list[CommitGroupStatus]:
        """Load every durable group, including terminal history, with path checks."""

        if not self.root.exists():
            return []
        result: list[CommitGroupStatus] = []
        for path in sorted(self.root.glob("*.json")):
            status = self._load_unlocked(path.stem)
            if status is None:
                raise CommitGroupIntegrityError(f"commit group disappeared during scan: {path.name}")
            if path.is_symlink() or self.path(status.group_id).resolve() != path.resolve():
                raise CommitGroupIntegrityError(f"commit group has an invalid artifact path: {path.name}")
            result.append(status)
        return result

    def recover_expired_consumers(self) -> list[tuple[str, str]]:
        recovered: list[tuple[str, str]] = []
        for status in self.pending():
            with self.group_lock(status.group_id):
                current = self._required_unlocked(status.group_id)
                changed = False
                if current.canonical_status == "running" and not self._lease_active(current.canonical_lease_expires_at):
                    current.canonical_status = "failed"
                    current.canonical_retryable = True
                    current.canonical_last_error = "canonical commit lease expired before completion"
                    current.canonical_attempt_id = ""
                    current.canonical_owner_pid = 0
                    current.canonical_lease_expires_at = ""
                    current.canonical_next_retry_at = _now()
                    changed = True
                for consumer, item in current.consumers.items():
                    if item.status == "running" and not self._lease_active(item.lease_expires_at):
                        item.status = "failed"
                        item.retryable = True
                        item.last_error = "consumer lease expired before completion"
                        item.attempt_id = ""
                        item.owner_pid = 0
                        item.lease_expires_at = ""
                        item.next_retry_at = _now()
                        recovered.append((status.group_id, consumer))
                        changed = True
                if changed:
                    current.updated_at = _now()
                    self._write(current)
        return recovered

    def recover_abandoned_leases(self) -> list[tuple[str, str]]:
        """Release leases whose recorded local process no longer exists.

        An unexpired lease owned by a live process is never stolen.  This lets
        startup recover immediately after a real process crash without
        weakening concurrent worker fencing.
        """

        recovered: list[tuple[str, str]] = []
        for status in self.pending():
            with self.group_lock(status.group_id):
                current = self._required_unlocked(status.group_id)
                changed = False
                if (
                    current.canonical_status == "running"
                    and current.canonical_owner_pid > 0
                    and not self._pid_alive(current.canonical_owner_pid)
                ):
                    current.canonical_status = "failed"
                    current.canonical_retryable = True
                    current.canonical_last_error = "canonical commit owner process exited"
                    current.canonical_attempt_id = ""
                    current.canonical_owner_pid = 0
                    current.canonical_lease_expires_at = ""
                    current.canonical_next_retry_at = _now()
                    recovered.append((status.group_id, "canonical"))
                    changed = True
                for consumer, item in current.consumers.items():
                    if item.status != "running" or item.owner_pid <= 0 or self._pid_alive(item.owner_pid):
                        continue
                    item.status = "failed"
                    item.retryable = True
                    item.last_error = "consumer owner process exited"
                    item.attempt_id = ""
                    item.owner_pid = 0
                    item.lease_expires_at = ""
                    item.next_retry_at = _now()
                    recovered.append((status.group_id, consumer))
                    changed = True
                if changed:
                    current.updated_at = _now()
                    self._write(current)
        return recovered

    def _required(self, group_id: str) -> CommitGroupStatus:
        status = self._load_unlocked(group_id)
        if status is None:
            raise KeyError(f"unknown commit group: {group_id}")
        return status

    def _required_unlocked(self, group_id: str) -> CommitGroupStatus:
        return self._required(group_id)

    def _consumer(self, status: CommitGroupStatus, consumer: str) -> ConsumerStatus:
        if consumer not in CONSUMERS:
            raise ValueError(f"unsupported commit group consumer: {consumer}")
        return status.consumers.setdefault(consumer, ConsumerStatus())

    def _write(self, status: CommitGroupStatus) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.path(status.group_id)
        payload = status.to_dict()
        payload["control_schema_version"] = "commit_group_control_v1"
        payload["control_digest"] = canonical_digest(payload)
        atomic_write_json(path, payload, artifact_root=self.artifact_root)

    def _load_unlocked(self, group_id: str) -> CommitGroupStatus | None:
        path = self.path(group_id)
        if not path.exists() and not path.is_symlink():
            return None
        try:
            if path.is_symlink():
                raise ValueError("commit group state cannot be a symbolic link")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("commit group state must be a JSON object")
            digest = payload.get("control_digest")
            core = {key: value for key, value in payload.items() if key != "control_digest"}
            if (
                payload.get("control_schema_version") != "commit_group_control_v1"
                or digest != canonical_digest(core)
                or payload.get("group_id") != group_id
            ):
                raise ValueError("commit group state digest or identity is corrupt")
            return CommitGroupStatus.from_dict(payload)
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            quarantine_control_file(
                self.artifact_root,
                path,
                kind="commit_group",
                error=exc,
                identifiers={"group_id": group_id},
            )
            raise CommitGroupIntegrityError("commit group state quarantined") from exc

    def _assert_attempt(self, item: ConsumerStatus, attempt_id: str | None) -> None:
        if attempt_id is not None and item.attempt_id != attempt_id:
            raise RuntimeError("commit group consumer attempt no longer owns the lease")

    def _lease_active(self, value: str) -> bool:
        if not value:
            return False
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return False
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc) > datetime.now(timezone.utc)

    def _retry_waiting(self, value: str) -> bool:
        return self._lease_active(value)

    def _pid_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _canonical_json(self, payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @contextmanager
    def group_lock(self, group_id: str) -> Iterator[None]:
        """Serialize one commit group across processes without blocking unrelated groups."""

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


def _now() -> str:
    return utc_now()
