from __future__ import annotations

from dataclasses import dataclass

from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_status import OperationStatus


@dataclass(frozen=True)
class ResolveResult:
    operation: ContextOperation
    resolved: bool
    reason: str = ""


class TargetResolver:
    def resolve(self, operation: ContextOperation) -> ResolveResult:
        if operation.target_uri:
            operation.status = OperationStatus.RESOLVED
            return ResolveResult(operation=operation, resolved=True, reason="target_uri provided")
        operation.status = OperationStatus.PENDING
        return ResolveResult(operation=operation, resolved=False, reason="target_uri required")
