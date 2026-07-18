"""操作提交里的冲突判断。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from memoryos.contextdb.model.context_type import ContextType
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus


class ConflictType(str, Enum):
    """负责 ConflictType 这部分逻辑。"""

    DUPLICATE = "duplicate"
    DELETE_OVERRIDES_UPDATE = "delete_overrides_update"
    POLICY_SUPPORT_CONSTRAINS_ACTION = "policy_support_constrains_action"
    DISABLED_POLICY_BLOCKS_REWARD = "disabled_policy_blocks_reward"
    SUPERSEDE_REQUIRES_TARGET = "supersede_requires_target"
    REWARD_PENALTY_MERGE = "reward_penalty_merge"


@dataclass(frozen=True)
class ConflictResult:
    """保存 ConflictResult 需要的这组数据。"""

    accepted: list[ContextOperation]
    rejected: list[ContextOperation]
    conflicts: list[dict]


@dataclass(frozen=True)
class PolicySupportMetadata:
    """Typed policy-support fields used by the generic operation plane."""

    support_anchor_kind: str
    policy_rule_type: str
    policy_rule_value: str
    related_action: str
    constrains_policy_uris: tuple[str, ...]


def operation_policy_support_metadata(operation: ContextOperation) -> PolicySupportMetadata:
    """Read only neutral support metadata; lexical text never grants policy authority."""

    context_object = operation.payload.get("context_object")
    metadata = dict(context_object.get("metadata", {}) or {}) if isinstance(context_object, dict) else {}
    related_action = str(
        metadata.get("related_action")
        or operation.payload.get("related_action")
        or operation.payload.get("action")
        or ""
    )
    constrained = metadata.get("constrains_policy_uris") or operation.payload.get("constrains_policy_uris") or []
    if not isinstance(constrained, list | tuple | set):
        constrained = []
    return PolicySupportMetadata(
        support_anchor_kind=str(metadata.get("support_anchor_kind") or ""),
        policy_rule_type=str(metadata.get("policy_rule_type") or operation.payload.get("policy_rule_type") or ""),
        policy_rule_value=str(
            metadata.get("policy_rule_value") or operation.payload.get("policy_rule_value") or ""
        ),
        related_action=related_action,
        constrains_policy_uris=tuple(str(item) for item in constrained),
    )


def _operation_tenant(operation: ContextOperation) -> str:
    context_object = operation.payload.get("context_object")
    object_tenant = context_object.get("tenant_id") if isinstance(context_object, dict) else None
    return str(operation.payload.get("tenant_id") or object_tenant or "default")


class ConflictResolver:
    """判断操作是否冲突，同时分开读取语义类型和存储元数据。"""

    def resolve(self, operations: list[ContextOperation]) -> ConflictResult:
        """结合当前状态解析出确定结果。"""

        preprocessed: list[ContextOperation] = []
        rejected: list[ContextOperation] = []
        conflicts: list[dict] = []
        for operation in operations:
            if operation.action == OperationAction.SUPERSEDE and not operation.target_uri:
                operation.status = OperationStatus.PENDING
                operation.payload = {**operation.payload, "reason": "pending_target_review"}
                preprocessed.append(operation)
                conflicts.append(
                    {
                        "type": ConflictType.SUPERSEDE_REQUIRES_TARGET.value,
                        "target": None,
                        "accepted": [operation.operation_id],
                        "rejected": [],
                        "reason": "supersede requires target review",
                    }
                )
                continue
            self._protect_disabled_auto_execute_reward(operation)
            preprocessed.append(operation)

        preprocessed, policy_conflicts = self._apply_policy_support_constraints(preprocessed)
        conflicts.extend(policy_conflicts)

        grouped: dict[tuple[str, str | None], list[ContextOperation]] = {}
        for operation in preprocessed:
            grouped.setdefault((operation.context_type.value, operation.target_uri), []).append(operation)
        accepted = []
        for key, items in grouped.items():
            chosen, dropped, conflict_type, reason = self._resolve_group(items)
            accepted.extend(chosen)
            rejected.extend(dropped)
            if dropped or conflict_type:
                conflicts.append(
                    {
                        "type": conflict_type.value if conflict_type else ConflictType.DUPLICATE.value,
                        "target": key,
                        "accepted": [item.operation_id for item in chosen],
                        "rejected": [item.operation_id for item in dropped],
                        "reason": reason,
                    }
                )
        return ConflictResult(accepted=accepted, rejected=rejected, conflicts=conflicts)

    def _resolve_group(
        self, operations: list[ContextOperation]
    ) -> tuple[list[ContextOperation], list[ContextOperation], ConflictType | None, str]:
        unique: list[ContextOperation] = []
        seen_actions: set[str] = set()
        duplicates: list[ContextOperation] = []
        for operation in operations:
            if operation.action.value in seen_actions:
                duplicates.append(operation)
                continue
            seen_actions.add(operation.action.value)
            unique.append(operation)
        if any(op.action == OperationAction.DELETE for op in unique):
            delete = [op for op in unique if op.action == OperationAction.DELETE][-1]
            return (
                [delete],
                [op for op in unique if op is not delete] + duplicates,
                ConflictType.DELETE_OVERRIDES_UPDATE,
                "delete overrides target mutations",
            )
        if any(op.action == OperationAction.SUPERSEDE for op in unique) and any(
            op.action == OperationAction.UPDATE for op in unique
        ):
            supersede = [op for op in unique if op.action == OperationAction.SUPERSEDE][-1]
            for update in [op for op in unique if op.action == OperationAction.UPDATE]:
                supersede.payload = {**update.payload, **supersede.payload}
                supersede.evidence = [*update.evidence, *supersede.evidence]
            dropped = [op for op in unique if op.action == OperationAction.UPDATE]
            keep = [op for op in unique if op.action != OperationAction.UPDATE]
            return keep, dropped + duplicates, None, "supersede merged update payload"
        if any(op.action == OperationAction.DISABLE for op in unique):
            disable = [op for op in unique if op.action == OperationAction.DISABLE][-1]
            keep = [
                op
                for op in unique
                if op.action
                in {
                    OperationAction.REWARD,
                    OperationAction.PENALIZE,
                    OperationAction.COOLDOWN,
                    OperationAction.SUPPRESS,
                }
            ]
            rejected = [op for op in unique if op not in [disable, *keep]]
            return [*keep, disable], rejected + duplicates, None, "disable preserved with policy-affecting operations"
        reward_ops = [op for op in unique if op.action == OperationAction.REWARD]
        penalty_ops = [op for op in unique if op.action == OperationAction.PENALIZE]
        if reward_ops and penalty_ops:
            merged = self._merge_reward_penalty(reward_ops[-1], penalty_ops[-1])
            dropped = [op for op in reward_ops + penalty_ops if op is not merged]
            keep = [op for op in unique if op.action not in {OperationAction.REWARD, OperationAction.PENALIZE}]
            return (
                [*keep, merged],
                dropped + duplicates,
                ConflictType.REWARD_PENALTY_MERGE,
                "reward and penalty merged by feedback strength",
            )
        if any(op.action == OperationAction.SUPPRESS for op in unique) and any(
            op.action == OperationAction.REWARD for op in unique
        ):
            suppress = [op for op in unique if op.action == OperationAction.SUPPRESS][-1]
            rejected = [op for op in unique if op is not suppress]
            return [suppress], rejected + duplicates, None, "suppress supersedes reward"
        return unique, duplicates, ConflictType.DUPLICATE if duplicates else None, "duplicates rejected"

    def _apply_policy_support_constraints(
        self, operations: list[ContextOperation]
    ) -> tuple[list[ContextOperation], list[dict]]:
        conflicts: list[dict] = []
        policy_rules = []
        for operation in operations:
            if operation.context_type != ContextType.ACTION_POLICY_SUPPORT:
                continue
            metadata = operation_policy_support_metadata(operation)
            if metadata.support_anchor_kind == "action_policy":
                policy_rules.append((operation, metadata))
        if not policy_rules:
            return operations, conflicts
        constrained = list(operations)
        for support_op, metadata in policy_rules:
            if metadata.policy_rule_type != "action_auto_execute" or metadata.policy_rule_value != "forbidden":
                continue
            if not metadata.constrains_policy_uris:
                continue
            for operation in constrained:
                if operation.context_type != ContextType.ACTION_POLICY:
                    continue
                if support_op.user_id != operation.user_id or _operation_tenant(support_op) != _operation_tenant(
                    operation
                ):
                    continue
                if operation.target_uri not in metadata.constrains_policy_uris:
                    continue
                context_object = operation.payload.get("context_object")
                object_metadata = (
                    dict(context_object.get("metadata", {}) or {}) if isinstance(context_object, dict) else {}
                )
                policy_action = str(operation.payload.get("action") or object_metadata.get("action") or "")
                if metadata.related_action and policy_action != metadata.related_action:
                    continue
                operation.payload = {
                    **operation.payload,
                    "auto_execute_allowed": False,
                    "status": "disabled_auto_execute",
                }
            conflicts.append(
                {
                    "type": ConflictType.POLICY_SUPPORT_CONSTRAINS_ACTION.value,
                    "target": support_op.target_uri,
                    "accepted": [support_op.operation_id],
                    "rejected": [],
                    "reason": "structured policy support constrains its exact action policy",
                }
            )
        return constrained, conflicts

    def _protect_disabled_auto_execute_reward(self, operation: ContextOperation) -> None:
        if operation.action != OperationAction.REWARD:
            return
        if (
            operation.payload.get("current_status") == "disabled_auto_execute"
            or operation.payload.get("auto_execute_allowed") is False
        ):
            operation.payload = {
                **operation.payload,
                "auto_execute_allowed": False,
                "do_not_restore_auto_execute": True,
            }

    def _merge_reward_penalty(self, reward: ContextOperation, penalty: ContextOperation) -> ContextOperation:
        reward_strength = abs(
            float(
                reward.payload.get(
                    "feedback_strength", reward.payload.get("reward", reward.payload.get("reward_value", 0.0))
                )
                or 0.0
            )
        )
        penalty_strength = abs(
            float(
                penalty.payload.get(
                    "feedback_strength", penalty.payload.get("penalty", penalty.payload.get("reward_value", 0.0))
                )
                or 0.0
            )
        )
        if penalty.payload.get("explicit_rule") or penalty_strength >= reward_strength:
            penalty.payload = {**reward.payload, **penalty.payload, "merged_from_reward": reward.operation_id}
            return penalty
        reward.payload = {**penalty.payload, **reward.payload, "merged_from_penalty": penalty.operation_id}
        return reward
