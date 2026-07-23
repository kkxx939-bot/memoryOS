"""以故障关闭方式打开本地耐久锁文件。"""

from __future__ import annotations

import os
import stat
from pathlib import Path


class FileLockIntegrityError(RuntimeError):
    """锁路径不安全，或已不再指向私有普通文件。"""


def open_private_lock(path: str | Path, *, root: str | Path) -> int:
    """打开一个锁，不跟随末级或根目录内的目录符号链接。"""

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
        # 某些可信路径构建器会先规范化主机路径别名再返回结果，
        # 例如 macOS 会把 /var 映射为 /private/var。这里接受这种精确的
        # 根目录规范化，但绝不解析不可信的子路径：下方基于
        # openat/O_NOFOLLOW 的逐级遍历仍必须识别并拒绝产物边界内的每个符号链接。
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
