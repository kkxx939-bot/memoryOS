"""操作提交里的审计写入器。"""

from __future__ import annotations

import json
from pathlib import Path

from memoryos.core.time import utc_now


class AuditWriter:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def record(self, user_id: str, event_type: str, payload: dict) -> Path:
        path = self.root / "system" / "audit" / f"{user_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        operation_id = str(payload.get("operation_id", ""))
        if operation_id and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if existing.get("payload", {}).get("operation_id") == operation_id:
                    return path
        record = {"created_at": utc_now(), "user_id": user_id, "event_type": event_type, "payload": payload}
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path
