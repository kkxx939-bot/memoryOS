"""Public commit APIs, loaded lazily to keep low-level proof code acyclic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memoryos.operations.commit.audit_writer import AuditWriter
    from memoryos.operations.commit.diff_writer import DiffWriter
    from memoryos.operations.commit.operation_coalescer import OperationCoalescer
    from memoryos.operations.commit.operation_committer import OperationCommitter
    from memoryos.operations.commit.redo_log import RedoLog

__all__ = ["AuditWriter", "DiffWriter", "OperationCoalescer", "OperationCommitter", "RedoLog"]


def __getattr__(name: str) -> Any:
    modules = {
        "AuditWriter": ("memoryos.operations.commit.audit_writer", "AuditWriter"),
        "DiffWriter": ("memoryos.operations.commit.diff_writer", "DiffWriter"),
        "OperationCoalescer": (
            "memoryos.operations.commit.operation_coalescer",
            "OperationCoalescer",
        ),
        "OperationCommitter": (
            "memoryos.operations.commit.operation_committer",
            "OperationCommitter",
        ),
        "RedoLog": ("memoryos.operations.commit.redo_log", "RedoLog"),
    }
    if name not in modules:
        raise AttributeError(name)
    module_name, attribute = modules[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
