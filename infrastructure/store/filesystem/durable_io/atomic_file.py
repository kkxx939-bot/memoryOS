"""Atomic creation of immutable durable files within a trusted root."""

from __future__ import annotations

import os
import stat
import uuid
from pathlib import Path

from infrastructure.store.filesystem.path_safety import DurablePathIntegrityError


class ImmutableArtifactConflictError(ValueError):
    """A create-only artifact identity is already bound to different bytes."""


def _open_control_parent(path: Path, artifact_root: str | Path) -> int:
    candidate = Path(path).expanduser().absolute()
    boundary = Path(artifact_root).expanduser().absolute()
    if boundary.is_symlink():
        raise DurablePathIntegrityError("artifact root cannot be a symbolic link")
    resolved_boundary = boundary.resolve()
    try:
        relative_parent = candidate.parent.relative_to(boundary)
    except ValueError:
        try:
            relative_parent = candidate.parent.relative_to(resolved_boundary)
            boundary = resolved_boundary
        except ValueError as exc:
            raise DurablePathIntegrityError("artifact path is outside its artifact root") from exc
    if any(part in {"", ".", ".."} for part in relative_parent.parts):
        raise DurablePathIntegrityError("artifact path contains an unsafe directory segment")
    boundary.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        boundary.chmod(0o700)
    except OSError:
        pass
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_descriptor = os.open(boundary, directory_flags)
    except OSError as exc:
        raise DurablePathIntegrityError("artifact root is not a safe directory") from exc
    try:
        for part in relative_parent.parts:
            try:
                os.mkdir(part, 0o700, dir_fd=directory_descriptor)
            except FileExistsError:
                pass
            os.fsync(directory_descriptor)
            try:
                child = os.open(part, directory_flags, dir_fd=directory_descriptor)
            except OSError as exc:
                raise DurablePathIntegrityError(
                    "artifact path cannot traverse a symbolic link or non-directory"
                ) from exc
            os.close(directory_descriptor)
            directory_descriptor = child
            try:
                os.fchmod(directory_descriptor, 0o700)
            except OSError:
                pass
        return directory_descriptor
    except BaseException:
        os.close(directory_descriptor)
        raise


def _read_regular_file_at(directory_descriptor: int, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except OSError as exc:
        raise ImmutableArtifactConflictError(
            "immutable artifact collision is unreadable or not a regular file"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ImmutableArtifactConflictError("immutable artifact collision is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def atomic_create_bytes(path: Path, encoded: bytes, *, artifact_root: str | Path) -> bool:
    """Create immutable bytes once; identical replay is a no-op."""

    try:
        parent_descriptor = _open_control_parent(path, artifact_root)
    except DurablePathIntegrityError as exc:
        raise ImmutableArtifactConflictError(str(exc)) from exc
    temporary_name = f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent_descriptor)
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:  # pragma: no cover - defensive OS contract.
                    raise OSError("immutable artifact write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            if _read_regular_file_at(parent_descriptor, path.name) != encoded:
                raise ImmutableArtifactConflictError(
                    "immutable artifact identity conflicts with different content"
                ) from None
            return False
        os.fsync(parent_descriptor)
        return True
    finally:
        try:
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        os.close(parent_descriptor)


__all__ = ["ImmutableArtifactConflictError", "atomic_create_bytes"]
