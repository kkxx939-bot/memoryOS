"""启动恢复产生的结构化结果。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RecoveryReport:
    """记录一次恢复是否完成以及各阶段的可观测结果。"""

    ready: bool
    details: dict[str, Any]
    reasons: tuple[str, ...] = ()


__all__ = ["RecoveryReport"]
