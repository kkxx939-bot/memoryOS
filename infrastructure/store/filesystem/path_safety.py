"""Lexical path integrity checks for durable local artifacts."""

from __future__ import annotations

import os
from pathlib import Path


class DurablePathIntegrityError(RuntimeError):
    """A durable artifact path escapes or traverses an in-boundary symlink."""


def require_safe_artifact_path(
    root: str | Path,
    path: str | Path,
    *,
    label: str,
) -> Path:
    """Validate one exact lexical path without following boundary aliases."""

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
    """Reject directory aliases anywhere in an authoritative tree.

    Quarantined payloads are evidence and may intentionally be symlinks moved
    without following their target.  The quarantine directory itself remains
    authoritative and must still be a real directory.  Leaf artifacts remain
    the responsibility of their typed validator, which can distinguish an
    authoritative proof from disposable projection state and repair the latter.
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
