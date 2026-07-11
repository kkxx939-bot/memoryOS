"""记忆系统里的记忆冷却。"""

from __future__ import annotations

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class MemoryCoolingPolicy:
    def cool(self, memory: ContextObject) -> ContextOperation:
        if memory.context_type != ContextType.MEMORY:
            raise ValueError("MemoryCoolingPolicy only accepts memory ContextObject")
        action = OperationAction.COMPRESS if memory.behavior_support_hotness > 0 else OperationAction.ARCHIVE
        return ContextOperation(
            user_id=str(memory.owner_user_id or ""),
            context_type=ContextType.MEMORY,
            action=action,
            target_uri=memory.uri,
            payload={
                "reason": (
                    "memory still has behavior or action-policy support"
                    if action == OperationAction.COMPRESS
                    else "memory is cold and unsupported"
                ),
                "delete": False,
            },
            confidence=1.0,
        )
