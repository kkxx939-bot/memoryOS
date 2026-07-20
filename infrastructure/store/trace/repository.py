"""召回轨迹 JSON 文件的唯一持久化实现。"""

from __future__ import annotations

import json
import os
import stat
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from infrastructure.store.filesystem.durable_io import atomic_write_json

DEFAULT_TRACE_MAX_AGE_SECONDS = 30 * 24 * 60 * 60
DEFAULT_TRACE_MAX_FILES = 10_000
DEFAULT_TRACE_MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_TRACE_RETENTION_SCAN_FILES = 20_000


class RecallTraceRepository:
    """安全保存、读取和清理单一租户的召回轨迹文件。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.root.chmod(0o700)
        except OSError as exc:
            raise PermissionError("recall trace directory permissions could not be secured") from exc

    @property
    def trace_root(self) -> Path:
        """返回轨迹根目录，供维护和安全清理使用。"""

        return self.root

    def save(self, trace_id: str, payload: Mapping[str, Any]) -> None:
        """原子写入已经由 Context 层清洗的轨迹结构。"""

        canonical_id = _canonical_trace_id(trace_id)
        value = dict(payload)
        if value.get("trace_id") != canonical_id:
            raise ValueError("recall trace payload identity does not match trace_id")
        atomic_write_json(
            self.root / f"{canonical_id}.json",
            value,
            artifact_root=self.root,
        )

    def read(self, trace_id: str) -> dict[str, Any]:
        """验证规范 UUID 和租户根后读取一个轨迹文件。"""

        canonical_id = _canonical_trace_id(trace_id)
        path = (self.root / f"{canonical_id}.json").resolve()
        try:
            path.relative_to(self.root)
        except ValueError:
            raise ValueError("trace path escapes its tenant root") from None
        if not path.is_file():
            raise FileNotFoundError(canonical_id)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("trace_id") != canonical_id:
            raise ValueError("recall trace is invalid")
        return value

    def prune(
        self,
        *,
        max_age_seconds: int = DEFAULT_TRACE_MAX_AGE_SECONDS,
        max_files: int = DEFAULT_TRACE_MAX_FILES,
        max_total_bytes: int = DEFAULT_TRACE_MAX_TOTAL_BYTES,
        now_epoch: float | None = None,
    ) -> dict[str, int]:
        """按时间、文件数和总大小清理旧轨迹。"""

        if max_age_seconds < 0 or max_files < 0 or max_total_bytes < 0:
            raise ValueError("recall trace retention limits must be non-negative")
        now = float(time.time() if now_epoch is None else now_epoch)
        entries: list[tuple[float, str, int, Path]] = []
        for entry in self.root.iterdir():
            name = entry.name
            if not name.endswith(".json"):
                continue
            trace_id = name.removesuffix(".json")
            try:
                _canonical_trace_id(trace_id)
            except ValueError:
                continue
            details = entry.lstat()
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
                raise PermissionError("recall trace retention encountered an unsafe trace entry")
            entries.append((float(details.st_mtime), name, int(details.st_size), entry))
            if len(entries) > MAX_TRACE_RETENTION_SCAN_FILES:
                raise RuntimeError("recall trace retention scan exceeded its hard bound")

        remove_names = {
            name
            for modified_at, name, _size, _entry in entries
            if max_age_seconds == 0 or now - modified_at > max_age_seconds
        }
        retained = [item for item in entries if item[1] not in remove_names]
        retained.sort(key=lambda item: (item[0], item[1]))
        retained_bytes = sum(item[2] for item in retained)
        while retained and (len(retained) > max_files or retained_bytes > max_total_bytes):
            _modified_at, name, size, _entry = retained.pop(0)
            remove_names.add(name)
            retained_bytes -= size

        deleted_bytes = 0
        for _modified_at, name, size, entry in entries:
            if name not in remove_names:
                continue
            resolved = entry.resolve(strict=True)
            try:
                resolved.relative_to(self.root)
            except ValueError:
                raise PermissionError("recall trace retention path escapes its tenant root") from None
            details = entry.lstat()
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
                raise PermissionError("recall trace retention encountered an unsafe trace entry")
            os.unlink(entry)
            deleted_bytes += size

        return {
            "scanned": len(entries),
            "deleted": len(remove_names),
            "deleted_bytes": deleted_bytes,
            "retained": len(entries) - len(remove_names),
            "retained_bytes": retained_bytes,
        }


def recall_trace_root(runtime_root: str | Path, tenant_id: str) -> Path:
    """根据运行根和租户返回唯一轨迹持久化目录。"""

    root = Path(runtime_root)
    return root / "recall-traces" if tenant_id == "default" else root / "tenants" / tenant_id / "recall-traces"


def _canonical_trace_id(trace_id: str) -> str:
    try:
        parsed = uuid.UUID(str(trace_id))
    except (AttributeError, TypeError, ValueError):
        raise ValueError("trace_id must be a canonical UUID") from None
    canonical_id = str(parsed)
    if canonical_id != str(trace_id):
        raise ValueError("trace_id must be a canonical UUID")
    return canonical_id


__all__ = [
    "DEFAULT_TRACE_MAX_AGE_SECONDS",
    "DEFAULT_TRACE_MAX_FILES",
    "DEFAULT_TRACE_MAX_TOTAL_BYTES",
    "RecallTraceRepository",
    "recall_trace_root",
]
