"""操作提交里的操作合并器。"""

from __future__ import annotations

from collections import defaultdict

from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus


class OperationCoalescer:
    """负责 OperationCoalescer 这部分逻辑。"""

    def coalesce(self, operations: list[ContextOperation]) -> list[ContextOperation]:
        grouped: dict[tuple[str, str | None], list[ContextOperation]] = defaultdict(list)
        passthrough = []
        for operation in operations:
            if operation.target_uri is None:
                passthrough.append(operation)
                continue
            grouped[operation.key()].append(operation)
        result = [self._coalesce_target(items) for items in grouped.values()]
        result.extend(passthrough)
        return [operation for operation in result if operation.status != OperationStatus.NOOP]

    def _coalesce_target(self, operations: list[ContextOperation]) -> ContextOperation:
        current = operations[0]
        for incoming in operations[1:]:
            current = self._merge_pair(current, incoming)
        return current

    def _merge_pair(self, first: ContextOperation, second: ContextOperation) -> ContextOperation:
        action_pair = (first.action, second.action)
        payload = self._merged_payload(first.payload, second.payload)
        evidence = [*first.evidence, *second.evidence]
        confidence = max(first.confidence, second.confidence)

        if action_pair == (OperationAction.ADD, OperationAction.UPDATE):
            return self._replace(first, action=OperationAction.ADD, payload=payload, evidence=evidence, confidence=confidence)
        if action_pair == (OperationAction.ADD, OperationAction.DELETE):
            return self._replace(second, status=OperationStatus.NOOP, payload={}, evidence=evidence, confidence=confidence)
        if action_pair == (OperationAction.UPDATE, OperationAction.DELETE):
            return self._replace(second, action=OperationAction.DELETE, payload=second.payload, evidence=evidence, confidence=confidence)
        if action_pair == (OperationAction.SUPERSEDE, OperationAction.UPDATE):
            return self._replace(first, action=OperationAction.SUPERSEDE, payload=payload, evidence=evidence, confidence=confidence)
        if {first.action, second.action} == {OperationAction.REWARD, OperationAction.PENALIZE}:
            return self._merge_reward_penalty(first, second)
        return self._replace(second, payload=payload, evidence=evidence, confidence=confidence)

    def _merge_reward_penalty(self, first: ContextOperation, second: ContextOperation) -> ContextOperation:
        payload = self._merged_payload(first.payload, second.payload)
        reward = float(first.payload.get("reward_delta", 0.0)) + float(second.payload.get("reward_delta", 0.0))
        penalty = float(first.payload.get("penalty_delta", 0.0)) + float(second.payload.get("penalty_delta", 0.0))
        payload["reward_delta"] = reward
        payload["penalty_delta"] = penalty
        action = OperationAction.REWARD if reward >= penalty else OperationAction.PENALIZE
        return self._replace(second, action=action, payload=payload, evidence=[*first.evidence, *second.evidence])

    def _replace(self, base: ContextOperation, **changes) -> ContextOperation:
        data = base.to_dict()
        data.update(changes)
        if "action" in data and isinstance(data["action"], OperationAction):
            data["action"] = data["action"].value
        if "context_type" in data and hasattr(data["context_type"], "value"):
            data["context_type"] = data["context_type"].value
        if "status" in data and isinstance(data["status"], OperationStatus):
            data["status"] = data["status"].value
        return ContextOperation.from_dict(data)

    def _merged_payload(self, first: dict, second: dict) -> dict:
        merged = dict(first)
        for key, value in second.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._merged_payload(merged[key], value)
            else:
                merged[key] = value
        return merged
