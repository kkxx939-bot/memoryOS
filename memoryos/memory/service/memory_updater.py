"""记忆系统里的记忆更新。"""

from __future__ import annotations

from memoryos.contextdb.model.context_type import ContextType
from memoryos.memory.lifecycle import MemoryCandidateLifecycle
from memoryos.memory.model.memory import Memory
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class MemoryUpdater:
    """负责 MemoryUpdater 这部分逻辑。"""

    def __init__(self) -> None:
        self.candidate_lifecycle = MemoryCandidateLifecycle()

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

    def confirm_candidate(self, *, user_id: str, candidate_uri: str, reason: str = "confirmed") -> ContextOperation:
        return self.candidate_lifecycle.confirm(user_id=user_id, candidate_uri=candidate_uri, reason=reason)

    def reject_candidate(self, *, user_id: str, candidate_uri: str, reason: str = "rejected") -> ContextOperation:
        return self.candidate_lifecycle.reject(user_id=user_id, candidate_uri=candidate_uri, reason=reason)

    def promote_candidate(self, *, user_id: str, candidate_uri: str, reason: str = "promoted") -> ContextOperation:
        return self.candidate_lifecycle.promote(user_id=user_id, candidate_uri=candidate_uri, reason=reason)
