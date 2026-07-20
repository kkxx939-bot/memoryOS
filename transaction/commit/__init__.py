"""提交层公开接口；延迟加载以避免底层校验证明形成循环依赖。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transaction.commit.operation_committer import OperationCommitter
    from transaction.commit.recovery import RecoveryResult, RecoveryService

__all__ = [
    "OperationCommitter",
    "RecoveryResult",
    "RecoveryService",
]


def __getattr__(name: str) -> Any:
    modules = {
        "OperationCommitter": (
            "transaction.commit.operation_committer",
            "OperationCommitter",
        ),
        "RecoveryResult": ("transaction.commit.recovery", "RecoveryResult"),
        "RecoveryService": ("transaction.commit.recovery", "RecoveryService"),
    }
    if name not in modules:
        raise AttributeError(name)
    module_name, attribute = modules[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
