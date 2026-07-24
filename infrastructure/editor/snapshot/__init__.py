"""读取旧版本并形成不可变快照的公共入口。"""

from infrastructure.editor.snapshot.model import (
    SnapshotBatch,
    SnapshotReadConfig,
    SnapshotState,
    VersionedSnapshot,
)
from infrastructure.editor.snapshot.reader import SnapshotReader, SnapshotReadLimitError

__all__ = [
    "SnapshotBatch",
    "SnapshotReadConfig",
    "SnapshotReadLimitError",
    "SnapshotReader",
    "SnapshotState",
    "VersionedSnapshot",
]
