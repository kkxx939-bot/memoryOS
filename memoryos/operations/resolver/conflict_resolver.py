from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from memoryos.contextdb.model.context_type import ContextType
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus


class ConflictType(str, Enum):
    DUPLICATE = "duplicate"
    DELETE_OVERRIDES_UPDATE = "delete_overrides_update"
    EXPLICIT_MEMORY_OVERRIDES_CANDIDATE = "explicit_memory_overrides_candidate"
    POLICY_MEMORY_CONSTRAINS_ACTION = "policy_memory_constrains_action"
    DISABLED_POLICY_BLOCKS_REWARD = "disabled_policy_blocks_reward"
    SUPERSEDE_REQUIRES_TARGET = "supersede_requires_target"
    REWARD_PENALTY_MERGE = "reward_penalty_merge"


@dataclass(frozen=True)
class ConflictResult:
    accepted: list[ContextOperation]
    rejected: list[ContextOperation]
    conflicts: list[dict]


class ConflictResolver:
    def resolve(self, operations: list[ContextOperation]) -> ConflictResult:
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

        preprocessed, policy_conflicts = self._apply_policy_memory_constraints(preprocessed)
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
        explicit, candidate = self._explicit_and_candidate(operations)
        if explicit and candidate:
            candidate.status = OperationStatus.REJECTED
            keep = [op for op in operations if op is not candidate]
            return keep, [candidate], ConflictType.EXPLICIT_MEMORY_OVERRIDES_CANDIDATE, "explicit memory overrides memory candidate"
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
            return [delete], [op for op in unique if op is not delete] + duplicates, ConflictType.DELETE_OVERRIDES_UPDATE, "delete overrides target mutations"
        if any(op.action == OperationAction.SUPERSEDE for op in unique) and any(op.action == OperationAction.UPDATE for op in unique):
            supersede = [op for op in unique if op.action == OperationAction.SUPERSEDE][-1]
            for update in [op for op in unique if op.action == OperationAction.UPDATE]:
                supersede.payload = {**update.payload, **supersede.payload}
                supersede.evidence = [*update.evidence, *supersede.evidence]
            dropped = [op for op in unique if op.action == OperationAction.UPDATE]
            keep = [op for op in unique if op.action != OperationAction.UPDATE]
            return keep, dropped + duplicates, None, "supersede merged update payload"
        if any(op.action == OperationAction.DISABLE for op in unique):
            disable = [op for op in unique if op.action == OperationAction.DISABLE][-1]
            keep = [op for op in unique if op.action in {OperationAction.REWARD, OperationAction.PENALIZE, OperationAction.COOLDOWN, OperationAction.SUPPRESS}]
            rejected = [op for op in unique if op not in [disable, *keep]]
            return [*keep, disable], rejected + duplicates, None, "disable preserved with policy-affecting operations"
        reward_ops = [op for op in unique if op.action == OperationAction.REWARD]
        penalty_ops = [op for op in unique if op.action == OperationAction.PENALIZE]
        if reward_ops and penalty_ops:
            merged = self._merge_reward_penalty(reward_ops[-1], penalty_ops[-1])
            dropped = [op for op in reward_ops + penalty_ops if op is not merged]
            keep = [op for op in unique if op.action not in {OperationAction.REWARD, OperationAction.PENALIZE}]
            return [*keep, merged], dropped + duplicates, ConflictType.REWARD_PENALTY_MERGE, "reward and penalty merged by feedback strength"
        if any(op.action == OperationAction.REJECT for op in unique) and any(op.action == OperationAction.CONFIRM for op in unique):
            confirm = [op for op in unique if op.action == OperationAction.CONFIRM][-1]
            rejected = [op for op in unique if op is not confirm]
            return [confirm], rejected + duplicates, None, "confirm supersedes candidate reject"
        if any(op.action == OperationAction.SUPPRESS for op in unique) and any(op.action == OperationAction.REWARD for op in unique):
            suppress = [op for op in unique if op.action == OperationAction.SUPPRESS][-1]
            rejected = [op for op in unique if op is not suppress]
            return [suppress], rejected + duplicates, None, "suppress supersedes reward"
        return unique, duplicates, ConflictType.DUPLICATE if duplicates else None, "duplicates rejected"

    def _explicit_and_candidate(self, operations: list[ContextOperation]) -> tuple[ContextOperation | None, ContextOperation | None]:
        explicit = None
        candidate = None
        for operation in operations:
            if operation.context_type != ContextType.MEMORY:
                continue
            memory_kind = operation.payload.get("memory_type") or operation.payload.get("memory_kind")
            context_object = operation.payload.get("context_object")
            if isinstance(context_object, dict):
                metadata = context_object.get("metadata", {})
                memory_kind = memory_kind or metadata.get("memory_type") or metadata.get("memory_kind")
            if memory_kind in {"explicit_memory", "policy_memory"}:
                explicit = operation
            if memory_kind == "memory_candidate":
                candidate = operation
        return explicit, candidate

    def _apply_policy_memory_constraints(self, operations: list[ContextOperation]) -> tuple[list[ContextOperation], list[dict]]:
        conflicts: list[dict] = []
        policy_memories = [
            op
            for op in operations
            if op.context_type == ContextType.MEMORY
            and (op.payload.get("memory_type") == "policy_memory" or op.payload.get("memory_kind") == "policy_memory")
        ]
        if not policy_memories:
            return operations, conflicts
        constrained = list(operations)
        for memory_op in policy_memories:
            text = " ".join(str(memory_op.payload.get(key, "")) for key in ("content", "rule", "title")).lower()
            if not any(token in text for token in ("不要自动", "禁止自动", "disable auto", "no auto", "do not automatically")):
                continue
            action = memory_op.payload.get("action")
            for operation in constrained:
                if operation.context_type != ContextType.ACTION_POLICY:
                    continue
                if action and operation.payload.get("action") and str(operation.payload.get("action")) != str(action):
                    continue
                operation.payload = {**operation.payload, "auto_execute_allowed": False, "status": "disabled_auto_execute"}
            conflicts.append(
                {
                    "type": ConflictType.POLICY_MEMORY_CONSTRAINS_ACTION.value,
                    "target": memory_op.target_uri,
                    "accepted": [memory_op.operation_id],
                    "rejected": [],
                    "reason": "policy memory constrains related action policy",
                }
            )
        return constrained, conflicts

    def _protect_disabled_auto_execute_reward(self, operation: ContextOperation) -> None:
        if operation.action != OperationAction.REWARD:
            return
        if operation.payload.get("current_status") == "disabled_auto_execute" or operation.payload.get("auto_execute_allowed") is False:
            operation.payload = {
                **operation.payload,
                "auto_execute_allowed": False,
                "do_not_restore_auto_execute": True,
            }

    def _merge_reward_penalty(self, reward: ContextOperation, penalty: ContextOperation) -> ContextOperation:
        reward_strength = abs(float(reward.payload.get("feedback_strength", reward.payload.get("reward", reward.payload.get("reward_value", 0.0))) or 0.0))
        penalty_strength = abs(float(penalty.payload.get("feedback_strength", penalty.payload.get("penalty", penalty.payload.get("reward_value", 0.0))) or 0.0))
        if penalty.payload.get("explicit_rule") or penalty_strength >= reward_strength:
            penalty.payload = {**reward.payload, **penalty.payload, "merged_from_reward": reward.operation_id}
            return penalty
        reward.payload = {**penalty.payload, **reward.payload, "merged_from_penalty": penalty.operation_id}
        return reward
