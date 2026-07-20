"""MemoryOS 的统一普通对象事务内核。"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transaction.commit import OperationCommitter, RecoveryResult, RecoveryService
    from transaction.model import ContextDiff, ContextOperation, OperationAction, OperationStatus

__all__ = [
    "ContextDiff",
    "ContextOperation",
    "OperationAction",
    "OperationCommitter",
    "OperationStatus",
    "RecoveryResult",
    "RecoveryService",
]


def __getattr__(name: str) -> Any:
    modules = {
        "ContextDiff": ("transaction.model", "ContextDiff"),
        "ContextOperation": ("transaction.model", "ContextOperation"),
        "OperationAction": ("transaction.model", "OperationAction"),
        "OperationCommitter": ("transaction.commit", "OperationCommitter"),
        "OperationStatus": ("transaction.model", "OperationStatus"),
        "RecoveryResult": ("transaction.commit", "RecoveryResult"),
        "RecoveryService": ("transaction.commit", "RecoveryService"),
    }
    if name not in modules:
        raise AttributeError(name)
    module_name, attribute = modules[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
