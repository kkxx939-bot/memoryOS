"""Ordinary Context operations for support objects."""

from __future__ import annotations

from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.support.model import SupportAnchor


class SupportAnchorUpdater:
    def add(self, anchor: SupportAnchor, *, evidence: list[dict] | None = None) -> ContextOperation:
        obj = anchor.to_context_object()
        return ContextOperation(
            user_id=anchor.user_id,
            context_type=obj.context_type,
            action=OperationAction.ADD,
            target_uri=anchor.uri,
            payload={"context_object": obj.to_dict(), "content": anchor.content},
            evidence=list(evidence or ()),
            confidence=anchor.confidence,
        )


__all__ = ["SupportAnchorUpdater"]
