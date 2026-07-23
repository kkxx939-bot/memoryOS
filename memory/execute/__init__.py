"""对外执行用户记忆命令、写入规划和审核流程的应用服务。"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AdoptResult",
    "MemoryCommandService",
    "MemoryEditReviewPreview",
    "MemoryEditReviewService",
]

_EXPORTS = {
    "AdoptResult": ("memory.execute.command_service", "AdoptResult"),
    "MemoryCommandService": ("memory.execute.command_service", "MemoryCommandService"),
    "MemoryEditReviewPreview": (
        "memory.execute.pending_review_service",
        "MemoryEditReviewPreview",
    ),
    "MemoryEditReviewService": (
        "memory.execute.pending_review_service",
        "MemoryEditReviewService",
    ),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value
