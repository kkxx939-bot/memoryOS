"""通用操作事务层唯一的操作归一化与冲突裁决入口。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from transaction.commit.domain_protocols import NoOperationDomainPolicy, OperationDomainPolicy
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_action import OperationAction
from transaction.model.operation_status import OperationStatus


class ConflictType(str, Enum):
    """通用操作冲突类型；具体领域拥有自己的冲突类型。"""

    DUPLICATE = "duplicate"
    ADD_DELETE_NOOP = "add_delete_noop"
    DELETE_OVERRIDES_UPDATE = "delete_overrides_update"


@dataclass(frozen=True)
class ConflictResult:
    """批量操作完成归一化后的确定结果。"""

    accepted: list[ContextOperation]
    rejected: list[ContextOperation]
    conflicts: list[dict]


class ConflictResolver:
    """统一处理通用冲突，并把领域冲突委托给注册的领域策略。"""

    def __init__(self, domain_policy: OperationDomainPolicy | None = None) -> None:
        self.domain_policy = domain_policy or NoOperationDomainPolicy()

    def resolve(self, operations: list[ContextOperation]) -> ConflictResult:
        """按目标分组，并为每组操作生成唯一确定的执行结果。"""

        preprocessed, conflicts = self.domain_policy.preprocess(operations)
        grouped: dict[tuple[str, str | None], list[ContextOperation]] = {}
        for operation in preprocessed:
            grouped.setdefault(operation.key(), []).append(operation)

        accepted: list[ContextOperation] = []
        rejected: list[ContextOperation] = []
        for key, items in grouped.items():
            domain = self.domain_policy.resolve_group(items)
            if domain is None:
                chosen, dropped, conflict_type, reason = self._resolve_generic_group(items)
            else:
                chosen = domain.accepted
                dropped = domain.rejected
                conflict_type = domain.conflict_type
                reason = domain.reason
            accepted.extend(chosen)
            rejected.extend(dropped)
            if dropped or conflict_type:
                conflicts.append(
                    {
                        "type": conflict_type or ConflictType.DUPLICATE.value,
                        "target": key,
                        "accepted": [item.operation_id for item in chosen],
                        "rejected": [item.operation_id for item in dropped],
                        "reason": reason,
                    }
                )
        return ConflictResult(accepted=accepted, rejected=rejected, conflicts=conflicts)

    def _resolve_generic_group(
        self,
        operations: list[ContextOperation],
    ) -> tuple[list[ContextOperation], list[ContextOperation], str | None, str]:
        add_positions = [index for index, operation in enumerate(operations) if operation.action == OperationAction.ADD]
        delete_positions = [
            index for index, operation in enumerate(operations) if operation.action == OperationAction.DELETE
        ]
        if add_positions and delete_positions and add_positions[0] < delete_positions[-1]:
            for operation in operations:
                operation.status = OperationStatus.NOOP
            return [], operations, ConflictType.ADD_DELETE_NOOP.value, "add followed by delete has no durable effect"

        if delete_positions:
            delete = operations[delete_positions[-1]]
            dropped = [operation for operation in operations if operation is not delete]
            return (
                [delete],
                dropped,
                ConflictType.DELETE_OVERRIDES_UPDATE.value,
                "delete overrides target mutations",
            )

        normalized, folded = self._fold_mutations(operations)
        supersedes = [operation for operation in normalized if operation.action == OperationAction.SUPERSEDE]
        updates = [operation for operation in normalized if operation.action == OperationAction.UPDATE]
        if supersedes and updates:
            supersede = supersedes[-1]
            for update in updates:
                supersede.payload = self._merged_payload(update.payload, supersede.payload)
                supersede.evidence = [*update.evidence, *supersede.evidence]
                supersede.confidence = max(update.confidence, supersede.confidence)
            keep = [operation for operation in normalized if operation.action != OperationAction.UPDATE]
            return keep, [*updates, *folded], None, "supersede merged update payload"

        duplicates: list[ContextOperation] = []
        unique: list[ContextOperation] = []
        seen_actions: set[OperationAction] = set()
        for operation in normalized:
            if operation.action in seen_actions:
                duplicates.append(operation)
                continue
            seen_actions.add(operation.action)
            unique.append(operation)
        dropped = [*folded, *duplicates]
        return (
            unique,
            dropped,
            ConflictType.DUPLICATE.value if dropped else None,
            "repeated operations were folded or rejected",
        )

    def _fold_mutations(
        self,
        operations: list[ContextOperation],
    ) -> tuple[list[ContextOperation], list[ContextOperation]]:
        """顺序折叠可组合写操作，不处理任何领域策略。"""

        result: list[ContextOperation] = []
        folded: list[ContextOperation] = []
        for operation in operations:
            if not result:
                result.append(operation)
                continue
            current = result[-1]
            pair = (current.action, operation.action)
            if pair == (OperationAction.ADD, OperationAction.UPDATE):
                current.payload = self._merged_payload(current.payload, operation.payload)
                current.evidence = [*current.evidence, *operation.evidence]
                current.confidence = max(current.confidence, operation.confidence)
                folded.append(operation)
                continue
            if pair in {
                (OperationAction.UPDATE, OperationAction.UPDATE),
                (OperationAction.MERGE, OperationAction.MERGE),
            }:
                current.payload = self._merged_payload(current.payload, operation.payload)
                current.evidence = [*current.evidence, *operation.evidence]
                current.confidence = max(current.confidence, operation.confidence)
                folded.append(operation)
                continue
            result.append(operation)
        return result, folded

    def _merged_payload(self, first: dict, second: dict) -> dict:
        merged = dict(first)
        for key, value in second.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merged_payload(merged[key], value)
            else:
                merged[key] = value
        return merged


__all__ = ["ConflictResolver", "ConflictResult", "ConflictType"]
