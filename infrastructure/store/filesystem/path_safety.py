"""耐久本地产物的词法路径完整性检查。"""

from __future__ import annotations

import os
from pathlib import Path


class DurablePathIntegrityError(RuntimeError):
    """耐久产物路径越出边界，或穿越边界内的符号链接。"""


def require_safe_artifact_path(
    root: str | Path,
    path: str | Path,
    *,
    label: str,
) -> Path:
    """校验一个精确的词法路径，不跟随边界别名。"""

    boundary = Path(root).expanduser().absolute()
    candidate = Path(path).expanduser().absolute()
    try:
        relative = candidate.relative_to(boundary)
    except ValueError as exc:
        raise DurablePathIntegrityError(f"{label} is outside its artifact root") from exc
    if boundary.is_symlink():
        raise DurablePathIntegrityError(f"{label} artifact root cannot be a symbolic link")
    current = boundary
    for part in relative.parts:
        if part in {"", ".", ".."}:
            raise DurablePathIntegrityError(f"{label} contains an unsafe path segment")
        current = current / part
        if current.is_symlink():
            raise DurablePathIntegrityError(f"{label} cannot traverse a symbolic link")
    return candidate


def validate_authoritative_tree(root: str | Path, *, label: str) -> int:
    """拒绝权威目录树中任何位置的目录别名。

    被隔离的载荷属于事实记录，可能是未跟随目标而直接移动的符号链接。
    隔离目录本身仍是权威目录，因此必须是真实目录。叶子产物仍由各自的
    类型校验器负责；类型校验器可以区分权威证明和可丢弃的投影状态，
    并修复后者。
    """

    boundary = Path(root).expanduser().absolute()
    if not boundary.exists():
        return 0
    require_safe_artifact_path(boundary.parent, boundary, label=label)
    checked = 0
    for directory, names, filenames in os.walk(boundary, followlinks=False):
        current = Path(directory)
        relative = current.relative_to(boundary)
        if relative.parts and relative.parts[0] == "quarantine":
            names[:] = []
            continue
        checked += len(names) + len(filenames)
        for name in names:
            candidate = current / name
            if candidate.is_symlink():
                raise DurablePathIntegrityError(f"{label} contains a symbolic link directory: {candidate}")
        if current == boundary:
            names[:] = [name for name in names if name != "quarantine"]
    return checked


__all__ = [
    "DurablePathIntegrityError",
    "require_safe_artifact_path",
    "validate_authoritative_tree",
]
