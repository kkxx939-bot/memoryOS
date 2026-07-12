"""适配器里的队列。"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, NoReturn

from memoryos.adapters.agent_hooks.sanitizer import sanitize_error_text
from memoryos.core.time import utc_now

_fcntl: Any
try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised by monkeypatch on POSIX CI.
    _fcntl = None

_msvcrt: Any
try:
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - exercised by monkeypatch on POSIX CI.
    _msvcrt = None

HOOK_ALLOWED_TOOLS = {
    "memoryos_search_context",
    "memoryos_assemble_context",
    "memoryos_commit_session",
    "memoryos_health",
}


class PendingQueueIntegrityError(RuntimeError):
    """A persisted hook queue cannot be trusted and has been quarantined."""


@dataclass
class PendingItem:
    event_id: str
    session_id: str
    adapter_id: str
    hook_name: str
    payload: dict[str, Any]
    tenant_id: str = "default"
    user_id: str = "default"
    retry_count: int = 0
    created_at: str = field(default_factory=utc_now)
    last_error: str = ""
    status: str = "pending"
    processing_until: str = ""


class PendingQueue:
    def __init__(
        self,
        path: str,
        *,
        tenant_id: str = "default",
        user_id: str = "default",
        max_retries: int = 3,
        processing_lease_seconds: int = 300,
    ) -> None:
        _validate_tenant_id(tenant_id)
        _validate_user_id(user_id)
        self.path = Path(path)
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.dead_letter_path = self.path.with_suffix(self.path.suffix + ".dead")
        self.max_retries = max(1, max_retries)
        self.processing_lease_seconds = max(1, processing_lease_seconds)

    def enqueue(self, item: PendingItem) -> bool:
        if not self._owns(item):
            raise PermissionError("pending hook item principal does not match the queue")
        queued = False

        def update(items: list[PendingItem]) -> list[PendingItem]:
            nonlocal queued
            if any(_item_key(existing) == _item_key(item) for existing in items):
                return items
            queued = True
            return [*items, item]

        self._locked_update(update)
        return queued

    def flush(self, client: Any, *, limit: int = 100) -> dict[str, Any]:
        flushed = 0
        failed = 0
        dead_lettered = 0
        claimed: list[PendingItem] = []

        def claim(items: list[PendingItem]) -> list[PendingItem]:
            nonlocal claimed
            now = _now_dt()
            lease_until = (now + timedelta(seconds=self.processing_lease_seconds)).isoformat()
            updated: list[PendingItem] = []
            claimed = []
            for item in items:
                if self._owns(item) and len(claimed) < limit and _is_flushable(item, now):
                    item.status = "processing"
                    item.processing_until = lease_until
                    claimed.append(item)
                updated.append(item)
            return updated

        self._locked_update(claim)
        retry_items: list[PendingItem] = []
        success_keys: set[tuple[str, str, str]] = set()
        dead_letter_items: list[PendingItem] = []
        for item in claimed:
            tool_name = str(item.payload.get("tool_name", "memoryos_commit_session"))
            arguments = item.payload.get("arguments", {})
            if tool_name not in HOOK_ALLOWED_TOOLS:
                item.retry_count += 1
                item.last_error = "DISALLOWED_HOOK_TOOL"
                item.status = "pending"
                item.processing_until = ""
                failed += 1
                dead_lettered += 1
                dead_letter_items.append(item)
                continue
            try:
                result = client.call_tool(tool_name, arguments)
                if isinstance(result, dict) and result.get("error"):
                    raise RuntimeError(str(result["error"].get("code", "MCP_ERROR")))
                flushed += 1
                success_keys.add(_item_key(item))
            except Exception as exc:
                item.retry_count += 1
                item.last_error = _safe_error(exc)
                item.status = "pending"
                item.processing_until = ""
                failed += 1
                if item.retry_count >= self.max_retries:
                    dead_lettered += 1
                    dead_letter_items.append(item)
                else:
                    retry_items.append(item)

        def finalize(items: list[PendingItem]) -> list[PendingItem]:
            retry_by_key = {_item_key(item): item for item in retry_items}
            dead_letter_keys = {_item_key(item) for item in dead_letter_items}
            remaining: list[PendingItem] = []
            seen_retry_keys: set[tuple[str, str, str]] = set()
            for item in items:
                key = _item_key(item)
                if key in success_keys or key in dead_letter_keys:
                    continue
                retry_item = retry_by_key.get(key)
                if retry_item is not None:
                    remaining.append(retry_item)
                    seen_retry_keys.add(key)
                    continue
                remaining.append(item)
            for item in retry_items:
                if _item_key(item) not in seen_retry_keys:
                    remaining.append(item)
            for item in dead_letter_items:
                self._append_dead_letter(item)
            return remaining

        if claimed:
            self._locked_update(finalize)
        remaining_count = len(self.list_items())
        return {"flushed": flushed, "failed": failed, "remaining": remaining_count, "dead_lettered": dead_lettered}

    def list_items(self) -> list[PendingItem]:
        return [item for item in self._read_items() if self._owns(item)]

    def mark_success(self, event_id: str) -> None:
        self._locked_update(
            lambda items: [item for item in items if not (self._owns(item) and item.event_id == event_id)]
        )

    def mark_failed(self, event_id: str, error: str) -> None:
        def update(items: list[PendingItem]) -> list[PendingItem]:
            for item in items:
                if self._owns(item) and item.event_id == event_id:
                    item.retry_count += 1
                    item.last_error = sanitize_error_text(str(error), max_text=300)
            return items

        self._locked_update(update)

    def _read_items(self) -> list[PendingItem]:
        if not self.path.exists():
            return []
        items: list[PendingItem] = []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            self._quarantine_corrupt_queue(exc)
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise TypeError("pending queue row must be a JSON object")
                items.append(PendingItem(**payload))
            except (json.JSONDecodeError, TypeError, ValueError):
                self._quarantine_corrupt_queue(
                    PendingQueueIntegrityError(
                        f"pending hook queue contains an invalid row at line {line_number}"
                    )
                )
        return items

    def _quarantine_corrupt_queue(self, error: BaseException) -> NoReturn:
        # Local import keeps the lightweight hook package out of the canonical
        # commit module's import cycle.
        from memoryos.operations.commit.quarantine import quarantine_control_file

        if self.path.exists():
            quarantine_control_file(
                self.path.parent,
                self.path,
                kind="hook_queue",
                error=error,
                identifiers={"tenant_id": self.tenant_id, "user_id": self.user_id},
            )
        raise PendingQueueIntegrityError("pending hook queue is corrupt and was quarantined") from error

    def _owns(self, item: PendingItem) -> bool:
        return item.tenant_id == self.tenant_id and item.user_id == self.user_id

    def _write_items(self, items: list[PendingItem]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        text = "\n".join(json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) for item in items)
        fd, tmp_name = tempfile.mkstemp(prefix=self.path.name, suffix=".tmp", dir=str(self.path.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            os.fchmod(fp.fileno(), 0o600)
            fp.write(text + ("\n" if text else ""))
        os.replace(tmp_name, self.path)
        os.chmod(self.path, 0o600)

    def _locked_update(self, update) -> int:  # noqa: ANN001
        lock_backend = _lock_backend()
        if lock_backend is None:
            raise RuntimeError("PendingQueue requires fcntl or msvcrt file locking")
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        if not self.lock_path.exists():
            self.lock_path.touch(mode=0o600)
        os.chmod(self.lock_path, 0o600)
        with self.lock_path.open("w", encoding="utf-8") as lock_file:
            _lock_file(lock_file, lock_backend)
            try:
                items = self._read_items()
                updated = update(items)
                self._write_items(updated)
                return len(updated)
            finally:
                _unlock_file(lock_file, lock_backend)

    def _append_dead_letter(self, item: PendingItem) -> None:
        self.dead_letter_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.dead_letter_path.parent, 0o700)
        with self.dead_letter_path.open("a", encoding="utf-8") as fp:
            os.chmod(self.dead_letter_path, 0o600)
            fp.write(json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) + "\n")
            fp.flush()
            os.fsync(fp.fileno())


def _safe_error(exc: Exception) -> str:
    return sanitize_error_text(str(exc) or exc.__class__.__name__, max_text=300)


def _lock_backend() -> str | None:
    if _fcntl is not None:
        return "fcntl"
    if _msvcrt is not None:
        return "msvcrt"
    return None


def _lock_file(lock_file: Any, backend: str) -> None:
    if backend == "fcntl":
        _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_EX)
        return
    _msvcrt.locking(lock_file.fileno(), _msvcrt.LK_LOCK, 1)


def _unlock_file(lock_file: Any, backend: str) -> None:
    if backend == "fcntl":
        _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_UN)
        return
    _msvcrt.locking(lock_file.fileno(), _msvcrt.LK_UNLCK, 1)


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _is_flushable(item: PendingItem, now: datetime) -> bool:
    if item.status != "processing":
        return True
    if not item.processing_until:
        return True
    try:
        return datetime.fromisoformat(item.processing_until) <= now
    except ValueError:
        return True


def _item_key(item: PendingItem) -> tuple[str, str, str]:
    return item.tenant_id, item.user_id, item.event_id


def _validate_tenant_id(tenant_id: str) -> None:
    if (
        not isinstance(tenant_id, str)
        or not tenant_id.strip()
        or tenant_id in {".", ".."}
        or "/" in tenant_id
        or "\\" in tenant_id
    ):
        raise ValueError("tenant_id must be one safe non-empty path segment")


def _validate_user_id(user_id: str) -> None:
    if (
        not isinstance(user_id, str)
        or not user_id.strip()
        or user_id in {".", ".."}
        or "/" in user_id
        or "\\" in user_id
    ):
        raise ValueError("user_id must be one safe non-empty path segment")
