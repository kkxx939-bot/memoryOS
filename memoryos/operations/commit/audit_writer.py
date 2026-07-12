"""操作提交里的审计写入器。"""

from __future__ import annotations

import json
import os
from pathlib import Path

from memoryos.core.ids import require_safe_path_segment
from memoryos.core.time import utc_now


class AuditWriter:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def record(self, user_id: str, event_type: str, payload: dict) -> Path:
        user_id = require_safe_path_segment(user_id, "user_id")
        path = self.root / "system" / "audit" / f"{user_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
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
        encoded = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return path
