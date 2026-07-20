"""把行为支撑证据投影为统一 Context 写操作。"""

from __future__ import annotations

from dataclasses import replace

from behavior.core.support import BehaviorSupportAnchor
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_type import ContextType
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_action import OperationAction


def behavior_support_to_context_object(anchor: BehaviorSupportAnchor) -> ContextObject:
    """生成只包含行为支撑字段的 Context 对象。"""

    return ContextObject(
        uri=anchor.uri,
        context_type=ContextType.BEHAVIOR_SUPPORT,
        title=anchor.title,
        owner_user_id=anchor.user_id,
        semantic_hotness=anchor.confidence,
        behavior_support_hotness=min(1.0, len(anchor.supporting_behavior_uris) * 0.15),
        metadata={
            "support_anchor_kind": "behavior",
            "anchor_key": anchor.anchor_key,
            "content": anchor.content,
            "supporting_behavior_uris": list(anchor.supporting_behavior_uris),
        },
        created_at=anchor.created_at,
        updated_at=anchor.updated_at,
    )


class BehaviorSupportWriter:
    """为行为支撑对象生成新增或更新操作，不直接访问 Store。"""

    def add(
        self,
        anchor: BehaviorSupportAnchor,
        *,
        evidence: list[dict] | None = None,
    ) -> ContextOperation:
        return self._operation(anchor, OperationAction.ADD, evidence=evidence)

    def update(
        self,
        anchor: BehaviorSupportAnchor,
        *,
        created_at: str,
        evidence: list[dict] | None = None,
    ) -> ContextOperation:
        preserved = replace(anchor, created_at=created_at)
        return self._operation(preserved, OperationAction.UPDATE, evidence=evidence)

    @staticmethod
    def _operation(
        anchor: BehaviorSupportAnchor,
        action: OperationAction,
        *,
        evidence: list[dict] | None,
    ) -> ContextOperation:
        obj = behavior_support_to_context_object(anchor)
        return ContextOperation(
            user_id=anchor.user_id,
            context_type=ContextType.BEHAVIOR_SUPPORT,
            action=action,
            target_uri=anchor.uri,
            payload={"context_object": obj.to_dict(), "content": anchor.content},
            evidence=list(evidence or ()),
            confidence=anchor.confidence,
        )


__all__ = [
    "BehaviorSupportWriter",
    "behavior_support_to_context_object",
]
