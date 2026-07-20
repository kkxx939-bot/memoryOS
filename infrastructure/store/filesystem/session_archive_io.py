"""会话归档共用的安全 JSON 与原子文件写入原语。"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from foundation.integrity import canonical_json
from memory.commit.evidence.errors import (
    EvidenceArchiveConflictError,
    EvidenceArchiveIntegrityError,
)


class SessionArchiveFileIO:
    """封装不可变对象、可替换 head 和私有目录的持久化规则。"""

    def __init__(self, root: Path) -> None:
        self.root = root

    def write_immutable_json(self, path: Path, payload: Any) -> None:
        """按 create-only 语义写入不可变 JSON，相同内容可幂等重放。"""

        self.write_create_only(path, canonical_json(payload).encode("utf-8"), compare_existing=True)

    def write_create_only(self, path: Path, payload: bytes, *, compare_existing: bool) -> None:
        """通过硬链接发布临时文件，确保已有目标永不被覆盖。"""

        self.secure_directory(path.parent)
        temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as handle:
                os.chmod(temporary, 0o600)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
                os.chmod(path, 0o600)
                self.fsync_directory(path.parent)
            except FileExistsError:
                if path.is_symlink():
                    raise EvidenceArchiveConflictError(
                        f"immutable evidence path cannot be a symbolic link: {path}"
                    ) from None
                if compare_existing and path.read_bytes() != payload:
                    raise EvidenceArchiveConflictError(
                        f"immutable evidence path contains different content: {path}"
                    ) from None
        finally:
            temporary.unlink(missing_ok=True)

    def write_head(self, path: Path, payload: dict[str, Any]) -> None:
        """原子替换可变控制 head，并拒绝符号链接目标。"""

        if path.is_symlink():
            raise EvidenceArchiveIntegrityError("session archive head cannot be a symbolic link")
        self.secure_directory(path.parent)
        temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                os.chmod(temporary, 0o600)
                handle.write(canonical_json(payload))
                handle.flush()
                os.fsync(handle.fileno())
            if path.is_symlink():
                raise EvidenceArchiveIntegrityError("session archive head cannot be a symbolic link")
            os.replace(temporary, path)
            os.chmod(path, 0o600)
            self.fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def write_bytes_atomic(self, path: Path, payload: bytes) -> None:
        """以私有权限原子安装一份可替换的输出文件。"""

        if path.is_symlink():
            raise EvidenceArchiveIntegrityError("session archive output cannot be a symbolic link")
        self.secure_directory(path.parent)
        temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as handle:
                os.chmod(temporary, 0o600)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if path.is_symlink():
                raise EvidenceArchiveIntegrityError("session archive output cannot be a symbolic link")
            os.replace(temporary, path)
            try:
                path.chmod(0o600)
            except OSError:
                pass
            self.fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def read_json(self, path: Path) -> Any:
        """读取归档 JSON，并把路径或编码问题统一映射为完整性错误。"""

        if path.is_symlink():
            raise EvidenceArchiveIntegrityError(f"evidence archive path cannot be a symbolic link: {path}")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise EvidenceArchiveIntegrityError(f"missing evidence archive object: {path}") from exc
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise EvidenceArchiveIntegrityError(f"invalid evidence archive JSON: {path}") from exc

    def secure_directory(self, directory: Path) -> None:
        """创建目录并把归档根目录以内的路径权限收紧为 0700。"""

        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        current = directory
        root = self.root.expanduser().resolve()
        while current == root or root in current.resolve().parents:
            try:
                current.chmod(0o700)
            except OSError:
                pass
            if current.resolve() == root:
                break
            current = current.parent

    @staticmethod
    def fsync_directory(directory: Path) -> None:
        """持久化目录项变化。"""

        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = ["SessionArchiveFileIO"]
