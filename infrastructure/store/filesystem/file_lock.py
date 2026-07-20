"""Fail-closed opening for local durable lock files."""

from __future__ import annotations

import os
import stat
from pathlib import Path


class FileLockIntegrityError(RuntimeError):
    """A lock path is unsafe or no longer names a private regular file."""


def open_private_lock(path: str | Path, *, root: str | Path) -> int:
    """Open one lock without following a final or in-root directory symlink."""

    lock_path = Path(path).expanduser()
    lexical_boundary = Path(root).expanduser()
    if not lock_path.is_absolute():
        lock_path = lock_path.absolute()
    if not lexical_boundary.is_absolute():
        lexical_boundary = lexical_boundary.absolute()
    if lexical_boundary.is_symlink():
        raise FileLockIntegrityError("lock directory cannot be a symbolic link")
    resolved_boundary = lexical_boundary.resolve()
    try:
        relative_parent = lock_path.parent.relative_to(lexical_boundary)
        boundary = lexical_boundary
    except ValueError:
        # Some trusted path builders normalize a host alias before returning
        # their result (macOS maps /var to /private/var).  Accept that exact
        # root normalization, but never resolve the untrusted child path: the
        # openat/O_NOFOLLOW walk below must still see and reject every symlink
        # inside the artifact boundary.
        try:
            relative_parent = lock_path.parent.relative_to(resolved_boundary)
            boundary = resolved_boundary
        except ValueError as exc:
            raise FileLockIntegrityError("lock path is outside its artifact root") from exc
    if any(part in {"", ".", ".."} for part in relative_parent.parts):
        raise FileLockIntegrityError("lock path contains an unsafe directory segment")
    boundary.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        boundary.chmod(0o700)
    except OSError:
        pass

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_descriptor = os.open(boundary, directory_flags)
    except OSError as exc:
        raise FileLockIntegrityError("lock root is not a safe directory") from exc
    try:
        for part in relative_parent.parts:
            try:
                os.mkdir(part, 0o700, dir_fd=directory_descriptor)
            except FileExistsError:
                pass
            try:
                child = os.open(part, directory_flags, dir_fd=directory_descriptor)
            except OSError as exc:
                raise FileLockIntegrityError("lock directory cannot be a symbolic link or non-directory") from exc
            os.close(directory_descriptor)
            directory_descriptor = child
            try:
                os.fchmod(directory_descriptor, 0o700)
            except OSError:
                pass

        nofollow = getattr(os, "O_NOFOLLOW", 0)
        create_flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | nofollow
        try:
            descriptor = os.open(
                lock_path.name,
                create_flags,
                0o600,
                dir_fd=directory_descriptor,
            )
        except FileExistsError:
            try:
                descriptor = os.open(
                    lock_path.name,
                    os.O_RDWR | nofollow,
                    dir_fd=directory_descriptor,
                )
            except OSError as exc:
                raise FileLockIntegrityError("lock file cannot be a symbolic link or unsafe file") from exc
        except OSError as exc:
            raise FileLockIntegrityError("lock file cannot be a symbolic link or unsafe file") from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise FileLockIntegrityError("lock path must be a regular file")
            os.fchmod(descriptor, 0o600)
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor
    finally:
        os.close(directory_descriptor)


__all__ = ["FileLockIntegrityError", "open_private_lock"]
