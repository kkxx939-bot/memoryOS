"""操作提交里的重做日志。"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from memoryos.operations.model.context_operation import ContextOperation


@dataclass(frozen=True)
class RedoEntry:
    operation: ContextOperation
    phase: str

    @property
    def operation_id(self) -> str:
        return self.operation.operation_id

    @property
    def target_uri(self) -> str | None:
        return self.operation.target_uri

    @property
    def user_id(self) -> str:
        return self.operation.user_id


class RedoLog:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.redo_dir = self.root / "system" / "redo"

    def begin(self, operation: ContextOperation, phase: str = "begin") -> Path:
        self.redo_dir.mkdir(parents=True, exist_ok=True)
        path = self.redo_dir / f"{operation.operation_id}.json"
        tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        payload = {**operation.to_dict(), "redo_phase": phase}
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return path

    def advance(self, operation: ContextOperation, phase: str) -> Path:
        return self.begin(operation, phase=phase)

    def commit(self, operation_id: str) -> None:
        path = self.redo_dir / f"{operation_id}.json"
        if path.exists():
            path.unlink()

    def pending(self) -> list[ContextOperation]:
        return [entry.operation for entry in self.pending_entries()]

    def pending_entries(self) -> list[RedoEntry]:
        if not self.redo_dir.exists():
            return []
        entries: list[RedoEntry] = []
        for path in sorted(self.redo_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            entries.append(RedoEntry(operation=ContextOperation.from_dict(payload), phase=str(payload.get("redo_phase", "started"))))
        return entries
