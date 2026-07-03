from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from memoryos.operations.model.context_operation import ContextOperation


class RedoLog:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.redo_dir = self.root / "system" / "redo"

    def begin(self, operation: ContextOperation) -> Path:
        self.redo_dir.mkdir(parents=True, exist_ok=True)
        path = self.redo_dir / f"{operation.operation_id}.json"
        tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(operation.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return path

    def commit(self, operation_id: str) -> None:
        path = self.redo_dir / f"{operation_id}.json"
        if path.exists():
            path.unlink()

    def pending(self) -> list[ContextOperation]:
        if not self.redo_dir.exists():
            return []
        return [
            ContextOperation.from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(self.redo_dir.glob("*.json"))
        ]
