"""事务冲突归并规则测试。"""

from __future__ import annotations

import unittest

from infrastructure.store.model.context.context_type import ContextType
from policy.action_policy.integration.conflict_policy import (
    POLICY_SUPPORT_CONSTRAINS_ACTION,
    ActionPolicyConflictPolicy,
)
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_action import OperationAction
from transaction.resolver.conflict_resolver import ConflictResolver, ConflictType


class ConflictResolverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = ConflictResolver(ActionPolicyConflictPolicy())

    def op(
        self,
        action: OperationAction,
        target: str | None = "uri",
        payload: dict | None = None,
        context_type: ContextType = ContextType.BEHAVIOR_CASE,
        *,
        user_id: str = "u1",
    ) -> ContextOperation:
        return ContextOperation(
            user_id=user_id,
            context_type=context_type,
            action=action,
            target_uri=target,
            payload=payload or {},
        )

    def policy_support(
        self,
        policy_uri: str,
        *,
        tenant_id: str = "default",
        related_action: str = "turn_on_ac",
        context_type: ContextType = ContextType.ACTION_POLICY_SUPPORT,
        user_id: str = "u1",
        metadata_overrides: dict | None = None,
    ) -> ContextOperation:
        metadata = {
            "support_anchor_kind": "action_policy",
            "policy_rule_type": "action_auto_execute",
            "policy_rule_value": "forbidden",
            "related_action": related_action,
            "constrains_policy_uris": [policy_uri],
            **(metadata_overrides or {}),
        }
        return self.op(
            OperationAction.ADD,
            target="memoryos://user/u1/support/action-policy/no-auto",
            payload={
                "tenant_id": tenant_id,
                "context_object": {"tenant_id": tenant_id, "metadata": metadata},
            },
            context_type=context_type,
            user_id=user_id,
        )

    def action_update(
        self,
        policy_uri: str,
        *,
        action: str = "turn_on_ac",
        tenant_id: str = "default",
        user_id: str = "u1",
    ) -> ContextOperation:
        return self.op(
            OperationAction.UPDATE,
            target=policy_uri,
            payload={
                "tenant_id": tenant_id,
                "action": action,
                "auto_execute_allowed": True,
            },
            context_type=ContextType.ACTION_POLICY,
            user_id=user_id,
        )

    def test_delete_overrides_update(self) -> None:
        result = self.resolver.resolve([self.op(OperationAction.UPDATE), self.op(OperationAction.DELETE)])
        self.assertEqual([op.action for op in result.accepted], [OperationAction.DELETE])
        self.assertEqual(result.conflicts[0]["type"], ConflictType.DELETE_OVERRIDES_UPDATE.value)

    def test_supersede_merges_update(self) -> None:
        update = self.op(OperationAction.UPDATE, payload={"a": 1})
        supersede = self.op(OperationAction.SUPERSEDE, payload={"b": 2})
        result = self.resolver.resolve([update, supersede])
        self.assertEqual(len(result.accepted), 1)
        self.assertEqual(result.accepted[0].action, OperationAction.SUPERSEDE)
        self.assertEqual(result.accepted[0].payload["a"], 1)
        self.assertEqual(result.accepted[0].payload["b"], 2)

    def test_structured_policy_support_constrains_exact_action_policy(self) -> None:
        policy_uri = "memoryos://user/u1/action_policies/hot/turn_on_ac"
        support = self.policy_support(policy_uri)
        action_update = self.action_update(policy_uri)

        result = self.resolver.resolve([support, action_update])

        self.assertFalse(action_update.payload["auto_execute_allowed"])
        self.assertEqual(action_update.payload["status"], "disabled_auto_execute")
        self.assertTrue(any(item["type"] == POLICY_SUPPORT_CONSTRAINS_ACTION for item in result.conflicts))

    def test_lexical_ordinary_text_has_no_policy_authority(self) -> None:
        policy_uri = "memoryos://user/u1/action_policies/hot/turn_on_ac"
        lexical_context = self.op(
            OperationAction.ADD,
            target="memoryos://user/u1/behavior_cases/no-auto",
            payload={"content": "不要自动执行", "action": "turn_on_ac"},
            context_type=ContextType.BEHAVIOR_CASE,
        )
        action_update = self.action_update(policy_uri)

        self.resolver.resolve([lexical_context, action_update])

        self.assertTrue(action_update.payload["auto_execute_allowed"])

    def test_policy_support_requires_structured_forbidden_rule(self) -> None:
        policy_uri = "memoryos://user/u1/action_policies/hot/turn_on_ac"
        malformed: tuple[dict[str, object], ...] = (
            {"support_anchor_kind": "behavior"},
            {"policy_rule_type": "other"},
            {"policy_rule_value": "allowed"},
            {"constrains_policy_uris": []},
        )
        for overrides in malformed:
            with self.subTest(overrides=overrides):
                action_update = self.action_update(policy_uri)
                result = self.resolver.resolve(
                    [self.policy_support(policy_uri, metadata_overrides=overrides), action_update]
                )
                self.assertTrue(action_update.payload["auto_execute_allowed"])
                self.assertFalse(any(item["type"] == POLICY_SUPPORT_CONSTRAINS_ACTION for item in result.conflicts))

    def test_policy_support_is_exact_uri_action_user_and_tenant_scoped(self) -> None:
        policy_uri = "memoryos://user/u1/action_policies/hot/turn_on_ac"
        cases = (
            (
                self.policy_support("memoryos://user/u1/action_policies/other/turn_on_ac"),
                self.action_update(policy_uri),
            ),
            (self.policy_support(policy_uri), self.action_update(policy_uri, action="turn_on_fan")),
            (self.policy_support(policy_uri, tenant_id="t1"), self.action_update(policy_uri, tenant_id="t2")),
            (self.policy_support(policy_uri, user_id="u2"), self.action_update(policy_uri, user_id="u1")),
        )
        for support, action_update in cases:
            with self.subTest(support=support.operation_id, action=action_update.operation_id):
                self.resolver.resolve([support, action_update])
                self.assertTrue(action_update.payload["auto_execute_allowed"])

    def test_disabled_auto_execute_reward_does_not_restore_auto_execute(self) -> None:
        reward = self.op(
            OperationAction.REWARD,
            target="policy",
            payload={"auto_execute_allowed": False},
            context_type=ContextType.ACTION_POLICY,
        )
        result = self.resolver.resolve([reward])
        self.assertTrue(result.accepted[0].payload["do_not_restore_auto_execute"])
        self.assertFalse(result.accepted[0].payload["auto_execute_allowed"])


if __name__ == "__main__":
    unittest.main()
