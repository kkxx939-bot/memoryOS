from __future__ import annotations

from memoryos.contextdb.model.context_type import ContextType
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class MemoryCandidateLifecycle:
    def confirm(self, *, user_id: str, candidate_uri: str, reason: str = "confirmed") -> ContextOperation:
        return ContextOperation(
            user_id=user_id,
            context_type=ContextType.MEMORY,
            action=OperationAction.CONFIRM,
            target_uri=candidate_uri,
            payload={"reason": reason},
        )

    def reject(self, *, user_id: str, candidate_uri: str, reason: str = "rejected") -> ContextOperation:
        return ContextOperation(
            user_id=user_id,
            context_type=ContextType.MEMORY,
            action=OperationAction.REJECT,
            target_uri=candidate_uri,
            payload={"reason": reason},
        )

    def promote(self, *, user_id: str, candidate_uri: str, reason: str = "promoted") -> ContextOperation:
        return self.confirm(user_id=user_id, candidate_uri=candidate_uri, reason=reason)
