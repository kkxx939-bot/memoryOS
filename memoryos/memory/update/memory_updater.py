from __future__ import annotations

from memoryos.contextdb.model.context_type import ContextType
from memoryos.memory.model.memory import Memory
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class MemoryUpdater:
    """Memory-only operation builder. Behavior and ActionPolicy updates live elsewhere."""

    def add_memory(self, memory: Memory, evidence: list[dict] | None = None) -> ContextOperation:
        return ContextOperation(
            user_id=memory.user_id,
            context_type=ContextType.MEMORY,
            action=OperationAction.ADD,
            target_uri=memory.uri,
            payload={"context_object": memory.to_context_object().to_dict(), "content": memory.content},
            evidence=evidence or [],
            confidence=memory.confidence,
        )

    def policy_rule(self, memory: Memory, evidence: list[dict] | None = None) -> ContextOperation:
        return self.add_memory(memory, evidence=evidence)
