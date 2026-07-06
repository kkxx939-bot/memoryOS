from __future__ import annotations

import fcntl
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from memoryos.core.time import utc_now

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


class PendingQueue:
    def __init__(self, path: str, *, max_retries: int = 3) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.dead_letter_path = self.path.with_suffix(self.path.suffix + ".dead")
        self.max_retries = max(1, max_retries)

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

        def update(items: list[PendingItem]) -> list[PendingItem]:
            nonlocal flushed, failed, dead_lettered
            remaining: list[PendingItem] = []
            for item in items[:limit]:
                tool_name = str(item.payload.get("tool_name", "memoryos_commit_session"))
                arguments = item.payload.get("arguments", {})
                if tool_name not in HOOK_ALLOWED_TOOLS:
                    item.retry_count += 1
                    item.last_error = "DISALLOWED_HOOK_TOOL"
                    failed += 1
                    dead_lettered += 1
                    self._append_dead_letter(item)
                    continue
                try:
                    result = client.call_tool(tool_name, arguments)
                    if isinstance(result, dict) and result.get("error"):
                        raise RuntimeError(str(result["error"].get("code", "MCP_ERROR")))
                    flushed += 1
                except Exception as exc:
                    item.retry_count += 1
                    item.last_error = str(exc)[:300] or exc.__class__.__name__
                    failed += 1
                    if item.retry_count >= self.max_retries:
                        dead_lettered += 1
                        self._append_dead_letter(item)
                    else:
                        remaining.append(item)
            remaining.extend(items[limit:])
            return remaining

        remaining_count = self._locked_update(update)
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
                    item.last_error = str(error)[:300]
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                items = self._read_items()
                updated = update(items)
                self._write_items(updated)
                return len(updated)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _append_dead_letter(self, item: PendingItem) -> None:
        self.dead_letter_path.parent.mkdir(parents=True, exist_ok=True)
        with self.dead_letter_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) + "\n")
