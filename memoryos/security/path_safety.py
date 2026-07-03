from __future__ import annotations

import re
from pathlib import Path

SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def validate_identifier(value: str, field_name: str) -> str:
    value = str(value or "")
    if not SAFE_IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field_name} must match ^[A-Za-z0-9_-]{{1,64}}$")
    return value


def safe_join(root: Path, rel_path: str | Path) -> Path:
    root = root.expanduser().resolve()
    path = (root / rel_path).expanduser().resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"path escapes memory root: {rel_path}")
    return path


def safe_relative_path(root: Path, path: Path) -> str:
    root = root.expanduser().resolve()
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path escapes memory root: {path}")
    return resolved.relative_to(root).as_posix()
