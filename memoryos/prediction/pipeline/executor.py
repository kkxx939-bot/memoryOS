from __future__ import annotations

from dataclasses import dataclass

from memoryos.prediction.model.prediction_result import PolicyDecision


@dataclass(frozen=True)
class ExecutionResult:
    mode: str
    action: str
    executed: bool
    reason: str


class Executor:
    def execute(self, decision: PolicyDecision) -> ExecutionResult:
        return ExecutionResult(
            mode=decision.mode,
            action=decision.action,
            executed=decision.mode == "execute" and decision.allowed,
            reason=decision.reason,
        )
