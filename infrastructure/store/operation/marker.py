"""普通操作幂等标记的文件持久化实现。"""

from __future__ import annotations

import json
from pathlib import Path

from foundation.ids import require_safe_path_segment
from infrastructure.store.filesystem.durable_io import atomic_create_json, atomic_write_json
from infrastructure.store.filesystem.durable_io.quarantine import quarantine_control_file


class OperationMarkerFileStore:
    """只负责标记文件寻址、原子创建、读取和更新。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.marker_dir = self.root / "system" / "operations"

    def path(self, operation_id: str) -> Path:
        key = require_safe_path_segment(operation_id, "operation_id")
        return self.marker_dir / f"{key}.json"

    def create(self, operation_id: str, payload: dict) -> bool:
        return atomic_create_json(
            self.path(operation_id),
            payload,
            artifact_root=self.root,
        )

    def read(self, path: Path) -> dict:
        if path.is_symlink():
            raise ValueError("operation marker cannot be a symbolic link")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("operation marker must be an object")
        return payload

    def paths(self) -> list[Path]:
        if not self.marker_dir.exists():
            return []
        return sorted(self.marker_dir.glob("*.json"))

    def replace(self, path: Path, payload: dict) -> None:
        if path.parent != self.marker_dir or path.suffix != ".json":
            raise ValueError("operation marker path is outside its store")
        atomic_write_json(path, payload, artifact_root=self.root)

    def quarantine(
        self,
        path: Path,
        error: BaseException,
        *,
        identifiers: dict[str, object],
    ) -> None:
        """隔离无法证明事务副作用的标记文件。"""

        quarantine_control_file(
            self.root,
            path,
            kind="operation_marker",
            error=error,
            identifiers=identifiers,
        )


__all__ = ["OperationMarkerFileStore"]
