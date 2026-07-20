"""Worker 调度器依赖的最小运行时表面。"""

from __future__ import annotations

from typing import Any, Protocol


class WorkerRuntime(Protocol):
    root: str
    runtime: Any


__all__ = ["WorkerRuntime"]
