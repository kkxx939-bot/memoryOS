"""普通操作事务所需控制存储的组合入口。

该对象只组合持久化实现，不承担提交顺序或领域判断。Runtime 与测试组合层在
创建事务提交器之前构造它，从而避免事务内核自行选择文件存储实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from infrastructure.store.operation.audit import AuditWriter
from infrastructure.store.operation.diff import DiffWriter
from infrastructure.store.operation.marker import OperationMarkerFileStore
from infrastructure.store.operation.redo import RedoLog


@dataclass(frozen=True)
class FileOperationControlStores:
    """一次普通操作事务共享的耐久控制存储。"""

    root: Path
    redo: RedoLog
    diff: DiffWriter
    audit: AuditWriter
    marker: OperationMarkerFileStore


def build_operation_control_stores(root: str | Path) -> FileOperationControlStores:
    """为一个已绑定租户的根目录创建文件控制存储。"""

    bound_root = Path(root)
    return FileOperationControlStores(
        root=bound_root,
        redo=RedoLog(bound_root),
        diff=DiffWriter(bound_root),
        audit=AuditWriter(bound_root),
        marker=OperationMarkerFileStore(bound_root),
    )


__all__ = ["FileOperationControlStores", "build_operation_control_stores"]
