"""Conversation 两层共用的安全、原子文件操作。"""

from __future__ import annotations

import os
import stat
import uuid
from datetime import date, datetime
from pathlib import Path

from foundation.ids import require_safe_path_segment


class ConversationFileIntegrityError(ValueError):
    """Conversation 文件路径或内容载体不满足完整性约束。"""


class ConversationFiles:
    """只负责 messages/summaries 文件定位和原子字节读写。"""

    _MAX_FILE_BYTES = 64 * 1024 * 1024

    def __init__(self, root: str | Path) -> None:
        requested = Path(root).expanduser().absolute()
        if requested.is_symlink():
            raise ConversationFileIntegrityError("conversation root cannot be a symbolic link")
        self.root = requested.resolve(strict=False)

    def initialize(self) -> Path:
        self._ensure_directory(self.root)
        self._ensure_directory(self.root / "messages")
        self._ensure_directory(self.root / "summaries")
        return self.root

    def path_for(
        self,
        branch: str,
        conversation_id: str,
        started_at: date | datetime,
        suffix: str,
    ) -> Path:
        if branch not in {"messages", "summaries"} or suffix not in {".jsonl", ".json"}:
            raise ConversationFileIntegrityError("unsupported conversation file branch")
        identifier = require_safe_path_segment(conversation_id, "conversation_id")
        if identifier != identifier.strip() or any(ord(character) < 32 for character in identifier):
            raise ConversationFileIntegrityError("conversation_id contains unsafe characters")
        logical_date = started_at.date() if isinstance(started_at, datetime) else started_at
        path = (
            self.root
            / branch
            / f"{logical_date.year:04d}"
            / f"{logical_date.month:02d}"
            / f"{logical_date.day:02d}"
            / f"{identifier}{suffix}"
        )
        self._require_inside_root(path)
        return path

    def write(self, path: Path, payload: bytes) -> Path:
        if len(payload) > self._MAX_FILE_BYTES:
            raise ConversationFileIntegrityError("conversation file exceeds its storage bound")
        self.initialize()
        self._ensure_directory(path.parent)
        if path.is_symlink():
            raise ConversationFileIntegrityError("conversation file cannot be a symbolic link")
        if path.exists():
            self._require_regular_file(path)
        temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("xb") as handle:
                os.chmod(temporary, 0o600)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if path.is_symlink():
                raise ConversationFileIntegrityError("conversation file cannot be a symbolic link")
            os.replace(temporary, path)
            os.chmod(path, 0o600)
            self._fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def read(self, path: Path) -> bytes:
        self._require_inside_root(path)
        self._require_regular_file(path)
        if path.stat().st_size > self._MAX_FILE_BYTES:
            raise ConversationFileIntegrityError("conversation file exceeds its read bound")
        return path.read_bytes()

    def _ensure_directory(self, directory: Path) -> None:
        self._require_inside_root(directory)
        relative = directory.relative_to(self.root)
        paths = (
            self.root,
            *(self.root / Path(*relative.parts[:index]) for index in range(1, len(relative.parts) + 1)),
        )
        for current in paths:
            if current.is_symlink():
                raise ConversationFileIntegrityError("conversation directory cannot be a symbolic link")
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                pass
            if not current.is_dir():
                raise ConversationFileIntegrityError("conversation directory path is not a directory")
            try:
                current.chmod(0o700)
            except OSError:
                pass

    def _require_inside_root(self, path: Path) -> None:
        resolved = path.resolve(strict=False)
        if resolved != self.root and self.root not in resolved.parents:
            raise ConversationFileIntegrityError("conversation path escapes its root")

    @staticmethod
    def _require_regular_file(path: Path) -> None:
        if path.is_symlink():
            raise ConversationFileIntegrityError("conversation file cannot be a symbolic link")
        try:
            metadata = path.stat()
        except FileNotFoundError:
            raise
        if not stat.S_ISREG(metadata.st_mode):
            raise ConversationFileIntegrityError("conversation path is not a regular file")

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = ["ConversationFileIntegrityError", "ConversationFiles"]
