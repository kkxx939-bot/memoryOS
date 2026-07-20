"""把 ActionPolicy 约束规则转换为统一 Context 写操作。"""

from __future__ import annotations

from dataclasses import replace

from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_type import ContextType
from policy.action_policy.model.policy_support_rule import PolicySupportRule
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_action import OperationAction


def policy_support_rule_to_context_object(rule: PolicySupportRule) -> ContextObject:
    """生成只包含策略约束字段的 Context 对象。"""

    return ContextObject(
        uri=rule.uri,
        context_type=ContextType.ACTION_POLICY_SUPPORT,
        title=rule.title,
        owner_user_id=rule.user_id,
        semantic_hotness=rule.confidence,
        metadata={
            "support_anchor_kind": "action_policy",
            "rule_key": rule.rule_key,
            "content": rule.content,
            "constrains_policy_uris": list(rule.constrains_policy_uris),
            "policy_rule_type": rule.policy_rule_type,
            "policy_rule_value": rule.policy_rule_value,
            "related_action": rule.related_action,
        },
        created_at=rule.created_at,
        updated_at=rule.updated_at,
    )


class PolicySupportWriter:
    """为策略约束规则生成新增或更新操作，不直接访问 Store。"""

    def add(
        self,
        rule: PolicySupportRule,
        *,
        evidence: list[dict] | None = None,
    ) -> ContextOperation:
        return self._operation(rule, OperationAction.ADD, evidence=evidence)

    def update(
        self,
        rule: PolicySupportRule,
        *,
        created_at: str,
        evidence: list[dict] | None = None,
    ) -> ContextOperation:
        preserved = replace(rule, created_at=created_at)
        return self._operation(preserved, OperationAction.UPDATE, evidence=evidence)

    @staticmethod
    def _operation(
        rule: PolicySupportRule,
        action: OperationAction,
        *,
        evidence: list[dict] | None,
    ) -> ContextOperation:
        obj = policy_support_rule_to_context_object(rule)
        return ContextOperation(
            user_id=rule.user_id,
            context_type=ContextType.ACTION_POLICY_SUPPORT,
            action=action,
            target_uri=rule.uri,
            payload={"context_object": obj.to_dict(), "content": rule.content},
            evidence=list(evidence or ()),
            confidence=rule.confidence,
        )


__all__ = [
    "PolicySupportWriter",
    "policy_support_rule_to_context_object",
]
