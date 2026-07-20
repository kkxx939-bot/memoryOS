"""ActionPolicy 自有的批量操作冲突规则。"""

from __future__ import annotations

from dataclasses import dataclass

from infrastructure.store.model.context.context_type import ContextType
from transaction.commit.domain_protocols import DomainConflictResolution
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_action import OperationAction
from transaction.resolver.conflict_resolver import ConflictType

POLICY_SUPPORT_CONSTRAINS_ACTION = "policy_support_constrains_action"
REWARD_PENALTY_MERGE = "reward_penalty_merge"


@dataclass(frozen=True)
class PolicySupportMetadata:
    """明确授权约束 ActionPolicy 的结构化支撑字段。"""

    support_anchor_kind: str
    policy_rule_type: str
    policy_rule_value: str
    related_action: str
    constrains_policy_uris: tuple[str, ...]


def operation_policy_support_metadata(operation: ContextOperation) -> PolicySupportMetadata:
    """只读取结构化约束字段，普通文本不能获得策略控制权。"""

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
        policy_rule_value=str(metadata.get("policy_rule_value") or operation.payload.get("policy_rule_value") or ""),
        related_action=related_action,
        constrains_policy_uris=tuple(str(item) for item in constrained),
    )


def _operation_tenant(operation: ContextOperation) -> str:
    context_object = operation.payload.get("context_object")
    object_tenant = context_object.get("tenant_id") if isinstance(context_object, dict) else None
    return str(operation.payload.get("tenant_id") or object_tenant or "default")


class ActionPolicyConflictPolicy:
    """集中处理策略反馈、禁用和结构化支撑约束。"""

    def preprocess(
        self,
        operations: list[ContextOperation],
    ) -> tuple[list[ContextOperation], list[dict]]:
        for operation in operations:
            self._protect_disabled_auto_execute_reward(operation)
        return self._apply_policy_support_constraints(operations)

    def resolve_group(
        self,
        operations: list[ContextOperation],
    ) -> DomainConflictResolution | None:
        if not operations or operations[0].context_type != ContextType.ACTION_POLICY:
            return None
        policy_actions = {
            OperationAction.REWARD,
            OperationAction.PENALIZE,
            OperationAction.COOLDOWN,
            OperationAction.SUPPRESS,
            OperationAction.DISABLE,
        }
        if not any(operation.action in policy_actions for operation in operations):
            return None

        unique: list[ContextOperation] = []
        duplicates: list[ContextOperation] = []
        seen_actions: set[OperationAction] = set()
        for operation in operations:
            if operation.action in seen_actions:
                duplicates.append(operation)
                continue
            seen_actions.add(operation.action)
            unique.append(operation)

        if any(operation.action == OperationAction.DISABLE for operation in unique):
            disable = [operation for operation in unique if operation.action == OperationAction.DISABLE][-1]
            feedback = [
                operation
                for operation in unique
                if operation.action
                in {
                    OperationAction.REWARD,
                    OperationAction.PENALIZE,
                    OperationAction.COOLDOWN,
                    OperationAction.SUPPRESS,
                }
            ]
            rejected = [operation for operation in unique if operation not in [disable, *feedback]]
            return DomainConflictResolution(
                accepted=[*feedback, disable],
                rejected=[*rejected, *duplicates],
                conflict_type=ConflictType.DUPLICATE.value if duplicates else None,
                reason="disable preserved with policy feedback operations",
            )

        reward_ops = [operation for operation in unique if operation.action == OperationAction.REWARD]
        penalty_ops = [operation for operation in unique if operation.action == OperationAction.PENALIZE]
        if reward_ops and penalty_ops:
            merged = self._merge_reward_penalty(reward_ops[-1], penalty_ops[-1])
            keep = [
                operation
                for operation in unique
                if operation.action not in {OperationAction.REWARD, OperationAction.PENALIZE}
            ]
            return DomainConflictResolution(
                accepted=[*keep, merged],
                rejected=[
                    *[operation for operation in [*reward_ops, *penalty_ops] if operation is not merged],
                    *duplicates,
                ],
                conflict_type=REWARD_PENALTY_MERGE,
                reason="reward and penalty merged by feedback strength",
            )

        if any(operation.action == OperationAction.SUPPRESS for operation in unique) and any(
            operation.action == OperationAction.REWARD for operation in unique
        ):
            suppress = [operation for operation in unique if operation.action == OperationAction.SUPPRESS][-1]
            return DomainConflictResolution(
                accepted=[suppress],
                rejected=[operation for operation in unique if operation is not suppress] + duplicates,
                conflict_type=None,
                reason="suppress supersedes reward",
            )

        return DomainConflictResolution(
            accepted=unique,
            rejected=duplicates,
            conflict_type=ConflictType.DUPLICATE.value if duplicates else None,
            reason="policy duplicates rejected",
        )

    def _apply_policy_support_constraints(
        self,
        operations: list[ContextOperation],
    ) -> tuple[list[ContextOperation], list[dict]]:
        conflicts: list[dict] = []
        policy_rules: list[tuple[ContextOperation, PolicySupportMetadata]] = []
        for operation in operations:
            if operation.context_type != ContextType.ACTION_POLICY_SUPPORT:
                continue
            metadata = operation_policy_support_metadata(operation)
            if metadata.support_anchor_kind == "action_policy":
                policy_rules.append((operation, metadata))
        for support_operation, metadata in policy_rules:
            if (
                metadata.policy_rule_type != "action_auto_execute"
                or metadata.policy_rule_value != "forbidden"
                or not metadata.constrains_policy_uris
            ):
                continue
            constrained: list[ContextOperation] = []
            for operation in operations:
                if operation.context_type != ContextType.ACTION_POLICY:
                    continue
                if support_operation.user_id != operation.user_id or _operation_tenant(
                    support_operation
                ) != _operation_tenant(operation):
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
                constrained.append(operation)
            if not constrained:
                continue
            conflicts.append(
                {
                    "type": POLICY_SUPPORT_CONSTRAINS_ACTION,
                    "target": support_operation.target_uri,
                    "accepted": [support_operation.operation_id, *[item.operation_id for item in constrained]],
                    "rejected": [],
                    "reason": "structured policy support constrains its exact action policy",
                }
            )
        return operations, conflicts

    @staticmethod
    def _protect_disabled_auto_execute_reward(operation: ContextOperation) -> None:
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

    @staticmethod
    def _merge_reward_penalty(
        reward: ContextOperation,
        penalty: ContextOperation,
    ) -> ContextOperation:
        reward_strength = abs(
            float(
                reward.payload.get(
                    "feedback_strength",
                    reward.payload.get("reward", reward.payload.get("reward_value", 0.0)),
                )
                or 0.0
            )
        )
        penalty_strength = abs(
            float(
                penalty.payload.get(
                    "feedback_strength",
                    penalty.payload.get("penalty", penalty.payload.get("reward_value", 0.0)),
                )
                or 0.0
            )
        )
        if penalty.payload.get("explicit_rule") or penalty_strength >= reward_strength:
            penalty.payload = {**reward.payload, **penalty.payload, "merged_from_reward": reward.operation_id}
            return penalty
        reward.payload = {**penalty.payload, **reward.payload, "merged_from_penalty": penalty.operation_id}
        return reward


__all__ = [
    "ActionPolicyConflictPolicy",
    "POLICY_SUPPORT_CONSTRAINS_ACTION",
    "PolicySupportMetadata",
    "REWARD_PENALTY_MERGE",
    "operation_policy_support_metadata",
]
