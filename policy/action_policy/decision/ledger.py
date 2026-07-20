"""ActionPolicy 在线决策审计端口。"""

from __future__ import annotations

from typing import Protocol

from policy.action_policy.decision.result import PredictionResult


class DecisionLedger(Protocol):
    """持久化实现需要满足的最小决策审计协议。"""

    def record(self, result: PredictionResult, *, tenant_id: str) -> object: ...


__all__ = ["DecisionLedger"]
