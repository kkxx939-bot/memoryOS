"""ActionPolicy 更新层中，把明确负向反馈转换为策略提交操作。"""

from __future__ import annotations

from foundation.ids import stable_hash
from infrastructure.store.model.context.context_type import ContextType
from policy.action_policy.model.policy_support_rule import PolicySupportRule
from policy.action_policy.update.policy_support_writer import PolicySupportWriter
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_action import OperationAction


class FeedbackCommitPlanner:
    """把用户明确禁止的行为规则转换为可提交的策略和支撑对象操作。"""

    def explicit_negative_rule_operations(
        self,
        *,
        user_id: str,
        policy_uri: str,
        explicit_rule: str,
        signal_type: str = "explicit_negative",
        evidence_uri: str = "",
        source_session_id: str | None = None,
        disable_policy: bool = True,
    ) -> list[ContextOperation]:
        if not explicit_rule:
            return []
        if not user_id or not policy_uri:
            raise ValueError("explicit policy rule requires user_id and policy_uri")
        related_action = policy_uri.rsplit("/", 1)[-1]
        digest = stable_hash([user_id, explicit_rule, policy_uri], length=16)
        policy_support = PolicySupportRule(
            uri=f"memoryos://user/{user_id}/support/action-policy/{digest}",
            user_id=user_id,
            title=explicit_rule[:48] or "policy support",
            content=explicit_rule,
            rule_key=digest,
            confidence=1.0,
            constrains_policy_uris=[policy_uri],
            policy_rule_type="action_auto_execute",
            policy_rule_value="forbidden",
            related_action=related_action,
        )
        support = PolicySupportWriter().add(
            policy_support,
            evidence=[{"type": "explicit_negative_rule", "uri": evidence_uri}],
        )
        support.source_session_id = source_session_id
        if not disable_policy:
            return [support]
        return [
            support,
            ContextOperation(
                user_id=user_id,
                context_type=ContextType.ACTION_POLICY,
                action=OperationAction.DISABLE,
                target_uri=policy_uri,
                payload={"auto_execute_allowed": False, "explicit_rule": explicit_rule},
                evidence=[{"type": signal_type, "uri": evidence_uri}],
                confidence=1.0,
                source_session_id=source_session_id,
            ),
        ]
