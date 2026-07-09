from __future__ import annotations

from memoryos.contextdb.model.context_type import ContextType
from memoryos.memory.model.memory import Memory
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class MemoryUpdater:
    """Memory-only operation builder. Behavior and ActionPolicy updates live elsewhere."""

    def add_memory(self, memory: Memory, evidence: list[dict] | None = None) -> ContextOperation:
        context_object = memory.to_context_object()
        metadata = dict(context_object.metadata)
        source = dict(memory.source or {})
        payload = {
            "context_object": context_object.to_dict(),
            "content": memory.content,
        }
        for key in (
            "memory_type",
            "admission",
            "retrieval_views",
            "merge_key",
            "schema_version",
        ):
            if key in metadata:
                payload[key] = metadata[key]
        if source.get("adapter_id"):
            payload["source_adapter_id"] = source["adapter_id"]
        if source.get("session_id"):
            payload["source_session_id"] = source["session_id"]
        if source.get("roles"):
            payload["source_roles"] = source["roles"]
        return ContextOperation(
            user_id=memory.user_id,
            context_type=ContextType.MEMORY,
            action=OperationAction.ADD,
            target_uri=memory.uri,
            payload=payload,
            evidence=evidence or [],
            confidence=memory.confidence,
        )

    def policy_rule(self, memory: Memory, evidence: list[dict] | None = None) -> ContextOperation:
        return self.add_memory(memory, evidence=evidence)
