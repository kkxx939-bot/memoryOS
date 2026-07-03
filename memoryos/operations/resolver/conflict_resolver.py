from __future__ import annotations

from dataclasses import dataclass

from memoryos.operations.model.context_operation import ContextOperation


@dataclass(frozen=True)
class ConflictResult:
    accepted: list[ContextOperation]
    rejected: list[ContextOperation]
    conflicts: list[dict]


class ConflictResolver:
    def resolve(self, operations: list[ContextOperation]) -> ConflictResult:
        seen: set[tuple[str, str | None, str]] = set()
        accepted = []
        rejected = []
        conflicts = []
        for operation in operations:
            key = (operation.context_type.value, operation.target_uri, operation.action.value)
            if key in seen:
                rejected.append(operation)
                conflicts.append({"operation_id": operation.operation_id, "reason": "duplicate operation"})
                continue
            seen.add(key)
            accepted.append(operation)
        return ConflictResult(accepted=accepted, rejected=rejected, conflicts=conflicts)
