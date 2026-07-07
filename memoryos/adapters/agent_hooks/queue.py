from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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


@dataclass
class PendingItem:
    event_id: str
    session_id: str
    adapter_id: str
    hook_name: str
    payload: dict[str, Any]
    retry_count: int = 0
    created_at: str = field(default_factory=utc_now)
    last_error: str = ""
    status: str = "pending"
    processing_until: str = ""


class PendingQueue:
    def __init__(self, path: str, *, max_retries: int = 3, processing_lease_seconds: int = 300) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.dead_letter_path = self.path.with_suffix(self.path.suffix + ".dead")
        self.max_retries = max(1, max_retries)
        self.processing_lease_seconds = max(1, processing_lease_seconds)

    def enqueue(self, item: PendingItem) -> bool:
        queued = False

        def update(items: list[PendingItem]) -> list[PendingItem]:
            nonlocal queued
            if any(existing.event_id == item.event_id for existing in items):
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
                if len(claimed) < limit and _is_flushable(item, now):
                    item.status = "processing"
                    item.processing_until = lease_until
                    claimed.append(item)
                updated.append(item)
            return updated

        self._locked_update(claim)
        retry_items: list[PendingItem] = []
        success_event_ids: set[str] = set()
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
                success_event_ids.add(item.event_id)
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
            retry_by_event_id = {item.event_id: item for item in retry_items}
            dead_letter_event_ids = {item.event_id for item in dead_letter_items}
            remaining: list[PendingItem] = []
            seen_retry_ids: set[str] = set()
            for item in items:
                if item.event_id in success_event_ids or item.event_id in dead_letter_event_ids:
                    continue
                retry_item = retry_by_event_id.get(item.event_id)
                if retry_item is not None:
                    remaining.append(retry_item)
                    seen_retry_ids.add(item.event_id)
                    continue
                remaining.append(item)
            for item in retry_items:
                if item.event_id not in seen_retry_ids:
                    remaining.append(item)
            for item in dead_letter_items:
                self._append_dead_letter(item)
            return remaining

        remaining_count = self._locked_update(finalize) if claimed else len(self.list_items())
        return {"flushed": flushed, "failed": failed, "remaining": remaining_count, "dead_lettered": dead_lettered}

    def list_items(self) -> list[PendingItem]:
        return self._read_items()

    def mark_success(self, event_id: str) -> None:
        self._locked_update(lambda items: [item for item in items if item.event_id != event_id])

    def mark_failed(self, event_id: str, error: str) -> None:
        def update(items: list[PendingItem]) -> list[PendingItem]:
            for item in items:
                if item.event_id == event_id:
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
        except OSError:
            return []
        for line in lines:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                items.append(PendingItem(**payload))
            except Exception:
                continue
        return items

    def _write_items(self, items: list[PendingItem]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) for item in items)
        fd, tmp_name = tempfile.mkstemp(prefix=self.path.name, suffix=".tmp", dir=str(self.path.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            fp.write(text + ("\n" if text else ""))
        os.replace(tmp_name, self.path)

    def _locked_update(self, update) -> int:  # noqa: ANN001
        lock_backend = _lock_backend()
        if lock_backend is None:
            raise RuntimeError("PendingQueue requires fcntl or msvcrt file locking")
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
        self.dead_letter_path.parent.mkdir(parents=True, exist_ok=True)
        with self.dead_letter_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) + "\n")


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
