from __future__ import annotations

from memoryos.behavior.model.behavior_pattern import BehaviorPattern
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class BehaviorPatternUpdater:
    def add_pattern(self, pattern: BehaviorPattern) -> ContextOperation:
        obj = pattern.to_context_object()
        return ContextOperation(
            user_id=pattern.user_id,
            context_type=obj.context_type,
            action=OperationAction.ADD,
            target_uri=obj.uri,
            payload={"context_object": obj.to_dict(), "content": pattern.to_context_object().metadata},
            evidence=[{"case_refs": pattern.case_refs}],
            confidence=pattern.confidence,
        )
