"""日志追踪里的审计日志。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from memoryos.core.clock import utc_now
from memoryos.security.path_safety import validate_identifier


class AuditLogger:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def record(self, user_id: str, event_type: str, payload: dict) -> dict:
        validate_identifier(user_id, "user_id")
        event = {
            "audit_id": self._audit_id(user_id, event_type, payload),
            "event_type": event_type,
            "user_id": user_id,
            "created_at": utc_now(),
            "payload": payload,
        }
        self._append(self._path(user_id), event)
        return event

    def list_events(self, user_id: str, event_type: str | None = None, limit: int | None = None) -> list[dict]:
        validate_identifier(user_id, "user_id")
        events = self._read_jsonl(self._path(user_id))
        if event_type:
            events = [event for event in events if event.get("event_type") == event_type]
        events.sort(key=lambda item: str(item.get("created_at", "")))
        return events[-limit:] if limit is not None else events

    def _path(self, user_id: str) -> Path:
        return self.root / "user" / user_id / "audit" / "audit_log.jsonl"

    def _append(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def _read_jsonl(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def _audit_id(self, user_id: str, event_type: str, payload: dict) -> str:
        material = json.dumps(
            {"user_id": user_id, "event_type": event_type, "payload": payload},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
