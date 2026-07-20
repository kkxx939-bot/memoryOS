"""Canonical, crash-safe JSON publication."""

from __future__ import annotations

import os
import stat
import uuid
from pathlib import Path
from typing import Any

from foundation.integrity.canonical_json import canonical_json
from infrastructure.store.filesystem.durable_io.atomic_file import _open_control_parent, atomic_create_bytes
from infrastructure.store.filesystem.path_safety import DurablePathIntegrityError


def atomic_create_json(path: Path, payload: dict[str, Any], *, artifact_root: str | Path) -> bool:
    """Create one canonical JSON artifact without ever replacing it."""

    return atomic_create_bytes(path, canonical_json(payload).encode("utf-8"), artifact_root=artifact_root)


def atomic_write_json(path: Path, payload: dict[str, Any], *, artifact_root: str | Path) -> None:
    """Publish one JSON file without exposing a partial control record."""

    try:
        parent_descriptor = _open_control_parent(path, artifact_root)
    except DurablePathIntegrityError as exc:
        raise ValueError(str(exc)) from exc
    temporary_name = f".{path.name}.{uuid.uuid4().hex}.tmp"
    encoded = canonical_json(payload).encode("utf-8")
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent_descriptor)
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:  # pragma: no cover - defensive OS contract.
                    raise OSError("JSON artifact write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            existing = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise ValueError("JSON control path cannot be a symbolic link or non-regular file")
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.fsync(parent_descriptor)
    finally:
        try:
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        os.close(parent_descriptor)


__all__ = ["atomic_create_json", "atomic_write_json"]
