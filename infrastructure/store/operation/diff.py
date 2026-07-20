"""操作事务差异记录的文件持久化实现。"""

from __future__ import annotations

import json
from pathlib import Path

from foundation.ids import require_safe_path_segment
from infrastructure.store.filesystem.durable_io import atomic_create_bytes


class DiffWriter:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, diff_id: str) -> Path:
        key = require_safe_path_segment(diff_id, "diff_id")
        return self.root / "system" / "diffs" / f"{key}.json"

    def write(self, payload: dict) -> Path:
        """原子创建不含语义正文的差异控制记录。"""

        diff_id = require_safe_path_segment(str(payload.get("diff_id") or ""), "diff_id")
        path = self.path(diff_id)
        if path.is_symlink():
            raise ValueError("diff control path cannot be a symbolic link")
        encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        atomic_create_bytes(path, encoded, artifact_root=self.root)
        return path

    def read(self, diff_id: str) -> dict:
        path = self.path(diff_id)
        if path.is_symlink():
            raise ValueError("diff control path cannot be a symbolic link")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("diff control record must be an object")
        return payload
