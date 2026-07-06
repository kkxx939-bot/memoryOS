from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from memoryos.core.time import utc_now


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
    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def enqueue(self, item: PendingItem) -> bool:
        items = self._read_items()
        if any(existing.event_id == item.event_id for existing in items):
            return False
        items.append(item)
        self._write_items(items)
        return True

    def flush(self, client: Any, *, limit: int = 100) -> dict[str, Any]:
        items = self._read_items()
        remaining: list[PendingItem] = []
        flushed = 0
        failed = 0
        for item in items[:limit]:
            tool_name = str(item.payload.get("tool_name", "memoryos_commit_session"))
            arguments = item.payload.get("arguments", {})
            try:
                result = client.call_tool(tool_name, arguments)
                if isinstance(result, dict) and result.get("error"):
                    raise RuntimeError(str(result["error"].get("code", "MCP_ERROR")))
                flushed += 1
            except Exception as exc:
                item.retry_count += 1
                item.last_error = exc.__class__.__name__
                failed += 1
                remaining.append(item)
        remaining.extend(items[limit:])
        self._write_items(remaining)
        return {"flushed": flushed, "failed": failed, "remaining": len(remaining)}

    def list_items(self) -> list[PendingItem]:
        return self._read_items()

    def mark_success(self, event_id: str) -> None:
        self._write_items([item for item in self._read_items() if item.event_id != event_id])

    def mark_failed(self, event_id: str, error: str) -> None:
        items = self._read_items()
        for item in items:
            if item.event_id == event_id:
                item.retry_count += 1
                item.last_error = str(error)[:300]
        self._write_items(items)

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
        self.path.write_text(text + ("\n" if text else ""), encoding="utf-8")
