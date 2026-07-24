"""目录 L0/L1 生成使用的受控数据模型。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from foundation.integrity import canonical_digest
from memory.model import MemoryDirectory


class MemorySemanticEntryKind(str, Enum):
    MEMORY = "memory"
    DIRECTORY = "directory"


@dataclass(frozen=True)
class MemorySemanticEntry:
    """一个目录中的直接 L2 文件或直接子目录摘要。"""

    name: str
    kind: MemorySemanticEntryKind
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("memory semantic entry name must be non-empty")
        if self.name != self.name.strip() or any(ord(character) < 32 for character in self.name):
            raise ValueError("memory semantic entry name contains unsafe characters")
        object.__setattr__(self, "kind", MemorySemanticEntryKind(self.kind))
        if not isinstance(self.content, str):
            raise TypeError("memory semantic entry content must be a string")


@dataclass(frozen=True)
class MemoryDirectorySnapshot:
    """生成一次目录概览所需的完整直接子项快照。"""

    directory: MemoryDirectory
    entries: tuple[MemorySemanticEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.directory, MemoryDirectory):
            raise TypeError("snapshot directory must be a MemoryDirectory")
        normalized = tuple(self.entries)
        if any(not isinstance(entry, MemorySemanticEntry) for entry in normalized):
            raise TypeError("snapshot entries must contain MemorySemanticEntry values")
        identities = [(entry.kind.value, entry.name) for entry in normalized]
        if len(identities) != len(set(identities)):
            raise ValueError("snapshot contains duplicate direct entries")
        object.__setattr__(self, "entries", normalized)

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "directory": list(self.directory.parts),
                "entries": [
                    {
                        "name": entry.name,
                        "kind": entry.kind.value,
                        "content": entry.content,
                    }
                    for entry in self.entries
                ],
            }
        )


class MemorySemanticRefreshStatus(str, Enum):
    WRITTEN = "written"
    DELETED = "deleted"
    UNCHANGED = "unchanged"
    MISSING = "missing"


@dataclass(frozen=True)
class MemorySemanticRefreshResult:
    """一个目录的 L0/L1 刷新结果。"""

    directory: MemoryDirectory
    status: MemorySemanticRefreshStatus
    source_digest: str = ""
    abstract_path: Path | None = None
    overview_path: Path | None = None


__all__ = [
    "MemoryDirectorySnapshot",
    "MemorySemanticEntry",
    "MemorySemanticEntryKind",
    "MemorySemanticRefreshResult",
    "MemorySemanticRefreshStatus",
]
