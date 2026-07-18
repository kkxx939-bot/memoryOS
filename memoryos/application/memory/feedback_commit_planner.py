"""动作策略里的反馈提交规划器。"""

from __future__ import annotations

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.action_policy.model.reward_signal import PenaltySignal
from memoryos.contextdb.model.context_type import ContextType
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.support import SupportAnchor, SupportAnchorKind, SupportAnchorUpdater


class FeedbackCommitPlanner:
    def explicit_negative_rule_operations(
        self,
        policy: ActionPolicy,
        signal: PenaltySignal,
    ) -> list[ContextOperation]:
        if not signal.explicit_rule:
            return []
        policy_support = SupportAnchor(
            uri=f"memoryos://user/{policy.user_id}/support/action-policy/{policy.scene_key}-{policy.action}-auto-execute",
            user_id=policy.user_id,
            title=f"Policy rule for {policy.action}",
            content=signal.explicit_rule,
            anchor_key=f"{policy.scene_key}-{policy.action}-auto-execute",
            kind=SupportAnchorKind.ACTION_POLICY,
            confidence=1.0,
            constrains_policy_uris=[policy.uri],
            policy_rule_type="action_auto_execute",
            policy_rule_value="forbidden",
            related_action=policy.action,
        )
        disable = ContextOperation(
            user_id=policy.user_id,
            context_type=ContextType.ACTION_POLICY,
            action=OperationAction.DISABLE,
            target_uri=policy.uri,
            payload={"auto_execute_allowed": False, "explicit_rule": signal.explicit_rule},
            evidence=[{"type": signal.signal_type, "uri": signal.evidence_uri}],
            confidence=1.0,
        )
        return [
            SupportAnchorUpdater().add(policy_support, evidence=[{"type": "explicit_negative_rule"}]),
            disable,
        ]
