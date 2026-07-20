"""行为模块里的行为模式更新器。"""

from __future__ import annotations

from behavior.core.model.behavior_pattern import BehaviorPattern
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_type import ContextType
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_action import OperationAction


def behavior_pattern_to_context_object(pattern: BehaviorPattern) -> ContextObject:
    """把纯行为模式映射为上下文存储对象。"""

    return ContextObject(
        uri=pattern.uri,
        context_type=ContextType.BEHAVIOR_PATTERN,
        title=f"BehaviorPattern {pattern.scene_key}",
        owner_user_id=pattern.user_id,
        hotness=pattern.hotness,
        behavior_support_hotness=pattern.confidence,
        metadata={
            "scene_key": pattern.scene_key,
            "trigger_conditions": pattern.trigger_conditions,
            "support_anchor_uri": pattern.support_anchor_uri,
            "case_refs": pattern.case_refs,
            "action_distribution": pattern.action_distribution,
            "opportunity": pattern.opportunity.__dict__,
            "status": pattern.status,
        },
        updated_at=pattern.updated_at,
    )


class BehaviorPatternUpdater:
    def add_pattern(self, pattern: BehaviorPattern) -> ContextOperation:
        obj = behavior_pattern_to_context_object(pattern)
        return ContextOperation(
            user_id=pattern.user_id,
            context_type=obj.context_type,
            action=OperationAction.ADD,
            target_uri=obj.uri,
            payload={"context_object": obj.to_dict(), "content": obj.metadata},
            evidence=[{"case_refs": pattern.case_refs}],
            confidence=pattern.confidence,
        )


__all__ = ["BehaviorPatternUpdater", "behavior_pattern_to_context_object"]
