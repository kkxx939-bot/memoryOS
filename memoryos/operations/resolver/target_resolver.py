"""操作提交里的目标解析器。"""

from __future__ import annotations

from dataclasses import dataclass, field

from memoryos.contextdb.store.source_store import IndexHit, IndexStore
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus


@dataclass(frozen=True)
class ResolveResult:
    operation: ContextOperation
    resolved: bool
    reason: str = ""
    candidates: list[IndexHit] = field(default_factory=list)


class TargetResolver:
    def __init__(self, index_store: IndexStore | None = None) -> None:
        self.index_store = index_store

    def resolve(self, operation: ContextOperation, user_id: str | None = None, limit: int = 5) -> ResolveResult:
        if operation.target_uri:
            operation.status = OperationStatus.RESOLVED
            return ResolveResult(operation=operation, resolved=True, reason="target_uri provided")

        if operation.action == OperationAction.ADD:
            operation.status = OperationStatus.RESOLVED
            return ResolveResult(operation=operation, resolved=True, reason="add operation creates its target")

        payload_target = operation.payload.get("policy_uri") or operation.payload.get("target_uri")
        if payload_target and operation.action in {
            OperationAction.REWARD,
            OperationAction.PENALIZE,
            OperationAction.COOLDOWN,
            OperationAction.SUPPRESS,
            OperationAction.DISABLE,
        }:
            operation.target_uri = str(payload_target)
            operation.status = OperationStatus.RESOLVED
            return ResolveResult(operation=operation, resolved=True, reason="target resolved from policy payload")

        candidates = self._candidate_targets(operation, user_id=user_id or operation.user_id, limit=limit)
        if candidates and candidates[0].score >= 0.75:
            operation.target_uri = candidates[0].uri
            operation.status = OperationStatus.RESOLVED
            return ResolveResult(operation=operation, resolved=True, reason="target resolved by semantic index", candidates=candidates)
        if candidates:
            operation.status = OperationStatus.PENDING
            operation.payload = {**operation.payload, "target_candidates": [hit.__dict__ for hit in candidates]}
            return ResolveResult(operation=operation, resolved=False, reason="target_review_required", candidates=candidates)
        operation.status = OperationStatus.PENDING
        return ResolveResult(operation=operation, resolved=False, reason="target_review_required")

    def _candidate_targets(self, operation: ContextOperation, user_id: str | None, limit: int) -> list[IndexHit]:
        if self.index_store is None:
            return []
        query = self._query_for(operation)
        if not query:
            return []
        return self.index_store.search(
            query,
            filters={"owner_user_id": user_id, "context_type": operation.context_type.value},
            limit=limit,
        )

    def _query_for(self, operation: ContextOperation) -> str:
        for key in ("query", "title", "content", "memory_anchor_uri"):
            value = operation.payload.get(key)
            if value:
                return str(value)
        scene_key = operation.payload.get("scene_key")
        action = operation.payload.get("action")
        if scene_key or action:
            return " ".join(str(item) for item in (scene_key, action) if item)
        return ""
