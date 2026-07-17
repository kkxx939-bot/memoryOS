"""Lazy compatibility exports for the historical prediction executor path."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memoryos.execution.action_executor import ActionExecutor, ExecutionResult, Executor

__all__ = ["ActionExecutor", "ExecutionResult", "Executor"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("memoryos.execution.action_executor"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
