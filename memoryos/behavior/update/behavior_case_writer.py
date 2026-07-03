from __future__ import annotations

from memoryos.behavior.model.behavior_case import BehaviorCase
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class BehaviorCaseWriter:
    def add_case(self, case: BehaviorCase) -> ContextOperation:
        uri = f"memoryos://user/{case.user_id}/behavior/cases/{case.scene_key}/{case.case_id}"
        obj = ContextObject(
            uri=uri,
            context_type=ContextType.BEHAVIOR_CASE,
            title=f"BehaviorCase {case.scene_key}",
            owner_user_id=case.user_id,
            metadata=case.to_dict(),
        )
        return ContextOperation(
            user_id=case.user_id,
            context_type=ContextType.BEHAVIOR_CASE,
            action=OperationAction.ADD,
            target_uri=uri,
            payload={"context_object": obj.to_dict(), "content": case.to_dict()},
            evidence=[{"case_id": case.case_id}],
            confidence=1.0,
        )
