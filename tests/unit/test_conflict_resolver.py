from __future__ import annotations

import unittest

from memoryos.contextdb.model.context_type import ContextType
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus
from memoryos.operations.resolver.conflict_resolver import ConflictResolver, ConflictType


class ConflictResolverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = ConflictResolver()

    def op(
        self,
        action: OperationAction,
        target: str | None = "uri",
        payload: dict | None = None,
        context_type: ContextType = ContextType.MEMORY,
    ) -> ContextOperation:
        return ContextOperation(
            user_id="u1", context_type=context_type, action=action, target_uri=target, payload=payload or {}
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

    def test_policy_memory_constrains_action_policy(self) -> None:
        policy_memory = self.op(
            OperationAction.ADD,
            target="rule",
            payload={
                "memory_type": "project_rule",
                "context_object": {
                    "metadata": {
                        "memory_kind": "policy_memory",
                        "state": "ACTIVE",
                        "canonical_rule_type": "action_auto_execute",
                        "related_action": "turn_on_ac",
                        "revisions": [{"value_fields": {"canonical_value": "forbidden"}}],
                    }
                },
            },
        )
        action_update = self.op(
            OperationAction.UPDATE,
            target="policy",
            payload={"action": "turn_on_ac"},
            context_type=ContextType.ACTION_POLICY,
        )
        result = self.resolver.resolve([policy_memory, action_update])
        self.assertFalse(action_update.payload["auto_execute_allowed"])
        self.assertEqual(action_update.payload["status"], "disabled_auto_execute")
        self.assertTrue(
            any(item["type"] == ConflictType.POLICY_MEMORY_CONSTRAINS_ACTION.value for item in result.conflicts)
        )

    def test_lexical_rule_without_active_structured_relation_cannot_modify_action_policy(self) -> None:
        memory = self.op(
            OperationAction.ADD,
            target="rule",
            payload={"memory_kind": "policy_memory", "content": "不要自动执行"},
        )
        action_update = self.op(
            OperationAction.UPDATE,
            target="policy",
            payload={"action": "turn_on_ac", "auto_execute_allowed": True},
            context_type=ContextType.ACTION_POLICY,
        )
        self.resolver.resolve([memory, action_update])
        self.assertTrue(action_update.payload["auto_execute_allowed"])

    def test_structured_policy_memory_cannot_constrain_another_scope(self) -> None:
        def scope(identifier: str) -> dict:
            return {"applicability": {"all_of": [{"namespace": "memoryos", "kind": "workspace", "id": identifier}]}}

        policy_memory = self.op(
            OperationAction.ADD,
            target="rule",
            payload={
                "memory_type": "project_rule",
                "context_object": {
                    "metadata": {
                        "memory_kind": "policy_memory",
                        "state": "ACTIVE",
                        "canonical_rule_type": "action_auto_execute",
                        "related_action": "turn_on_ac",
                        "scope": scope("workspace-a"),
                        "revisions": [{"value_fields": {"canonical_value": "forbidden"}}],
                    }
                },
            },
        )
        action_update = self.op(
            OperationAction.UPDATE,
            target="policy",
            payload={
                "action": "turn_on_ac",
                "auto_execute_allowed": True,
                "scope": scope("workspace-b"),
            },
            context_type=ContextType.ACTION_POLICY,
        )
        self.resolver.resolve([policy_memory, action_update])
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

    def test_supersede_without_target_stays_pending(self) -> None:
        supersede = self.op(OperationAction.SUPERSEDE, target=None)
        result = self.resolver.resolve([supersede])
        self.assertEqual(result.accepted[0].status, OperationStatus.PENDING)
        self.assertTrue(any(item["type"] == ConflictType.SUPERSEDE_REQUIRES_TARGET.value for item in result.conflicts))


if __name__ == "__main__":
    unittest.main()
