"""事务内核使用的控制记录类型与持久化端口。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from transaction.model.context_operation import ContextOperation


class RedoIntegrityError(RuntimeError):
    """耐久 Redo 阶段与当前事务副作用不一致。"""


class RedoControlFileError(RedoIntegrityError):
    """一个或多个损坏的 Redo 控制文件已被隔离。"""

    def __init__(self, records: Sequence[object]) -> None:
        super().__init__("corrupt redo control file quarantined")
        self.records = tuple(records)


@dataclass(frozen=True)
class RedoEntry:
    """恢复普通事务所需的最小耐久意图。"""

    operation: ContextOperation
    phase: str
    source_effect: dict | None = None
    relation_manifest: dict | None = None

    @property
    def operation_id(self) -> str:
        return self.operation.operation_id

    @property
    def target_uri(self) -> str | None:
        return self.operation.target_uri

    @property
    def user_id(self) -> str:
        return self.operation.user_id


class RedoStore(Protocol):
    redo_dir: Path

    def begin(
        self,
        operation: ContextOperation,
        phase: str = "begin",
        *,
        source_effect: dict | None = None,
        relation_manifest: dict | None = None,
    ) -> Path: ...

    def advance(
        self,
        operation: ContextOperation,
        phase: str,
        *,
        source_effect: dict | None = None,
        relation_manifest: dict | None = None,
    ) -> Path: ...

    def commit(self, operation_id: str) -> None: ...

    def pending_entries(self) -> list[RedoEntry]: ...

    def quarantine(self, operation_id: str, error: BaseException) -> bool: ...


class DiffStore(Protocol):
    def path(self, diff_id: str) -> Path: ...

    def write(self, payload: dict) -> Path: ...

    def read(self, diff_id: str) -> dict: ...


class AuditStore(Protocol):
    def record(self, user_id: str, event_type: str, payload: dict) -> Path: ...


class MarkerStore(Protocol):
    def path(self, operation_id: str) -> Path: ...

    def create(self, operation_id: str, payload: dict) -> bool: ...

    def read(self, path: Path) -> dict: ...

    def paths(self) -> list[Path]: ...

    def replace(self, path: Path, payload: dict) -> None: ...

    def quarantine(
        self,
        path: Path,
        error: BaseException,
        *,
        identifiers: dict[str, object],
    ) -> None: ...


class OperationControlStores(Protocol):
    """由外部组合层注入事务内核的耐久控制存储集合。"""

    @property
    def root(self) -> Path: ...

    @property
    def redo(self) -> RedoStore: ...

    @property
    def diff(self) -> DiffStore: ...

    @property
    def audit(self) -> AuditStore: ...

    @property
    def marker(self) -> MarkerStore: ...


__all__ = [
    "AuditStore",
    "DiffStore",
    "MarkerStore",
    "OperationControlStores",
    "RedoControlFileError",
    "RedoEntry",
    "RedoIntegrityError",
    "RedoStore",
]
