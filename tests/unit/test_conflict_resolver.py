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
        global_scope = {
            "applicability": {
                "all_of": [{"namespace": "memoryos", "kind": "global", "id": "tenant"}]
            }
        }
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
                        "scope": global_scope,
                        "revisions": [{"value_fields": {"canonical_value": "forbidden"}}],
                    }
                },
            },
        )
        action_update = self.op(
            OperationAction.UPDATE,
            target="policy",
            payload={"action": "turn_on_ac", "scope": global_scope},
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

    def test_structured_policy_memory_cannot_constrain_same_asset_id_under_another_parent(self) -> None:
        def scope(parent: str) -> dict:
            return {
                "applicability": {
                    "all_of": [
                        {
                            "namespace": "memoryos",
                            "kind": "asset",
                            "id": "camera",
                            "parent_path": [parent],
                        }
                    ]
                }
            }

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
                        "related_action": "capture",
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
                "action": "capture",
                "auto_execute_allowed": True,
                "scope": scope("workspace-b"),
            },
            context_type=ContextType.ACTION_POLICY,
        )

        self.resolver.resolve([policy_memory, action_update])

        self.assertTrue(action_update.payload["auto_execute_allowed"])

    def test_malformed_all_of_cannot_be_dropped_to_make_policy_rule_global(self) -> None:
        malformed_scope = {
            "applicability": {
                "all_of": [
                    {"namespace": "memoryos", "kind": "workspace", "id": "w1"},
                    {"namespace": "memoryos", "kind": "location"},
                ]
            }
        }
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
                        "related_action": "capture",
                        "scope": malformed_scope,
                        "revisions": [{"value_fields": {"canonical_value": "forbidden"}}],
                    }
                },
            },
        )
        action_update = self.op(
            OperationAction.UPDATE,
            target="policy",
            payload={
                "action": "capture",
                "auto_execute_allowed": True,
                "scope": {
                    "applicability": {
                        "all_of": [{"namespace": "memoryos", "kind": "workspace", "id": "w1"}]
                    }
                },
            },
            context_type=ContextType.ACTION_POLICY,
        )

        self.resolver.resolve([policy_memory, action_update])

        self.assertTrue(action_update.payload["auto_execute_allowed"])

    def test_empty_scope_cannot_be_promoted_to_global_policy(self) -> None:
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
                        "related_action": "capture",
                        "scope": {"applicability": {"all_of": []}},
                        "revisions": [{"value_fields": {"canonical_value": "forbidden"}}],
                    }
                },
            },
        )
        action_update = self.op(
            OperationAction.UPDATE,
            target="policy",
            payload={
                "action": "capture",
                "auto_execute_allowed": True,
                "scope": {
                    "applicability": {
                        "all_of": [{"namespace": "memoryos", "kind": "workspace", "id": "w1"}]
                    }
                },
            },
            context_type=ContextType.ACTION_POLICY,
        )

        self.resolver.resolve([policy_memory, action_update])

        self.assertTrue(action_update.payload["auto_execute_allowed"])

    def test_explicit_falsy_metadata_scope_never_falls_back_to_top_level_scope(self) -> None:
        workspace_scope = {
            "applicability": {
                "all_of": [{"namespace": "memoryos", "kind": "workspace", "id": "w1"}]
            }
        }
        malformed_scopes: tuple[list[object] | dict[str, object] | None, ...] = ([], {}, None)
        for malformed_scope in malformed_scopes:
            with self.subTest(malformed_scope=malformed_scope):
                policy_memory = self.op(
                    OperationAction.ADD,
                    target="rule",
                    payload={
                        "memory_type": "project_rule",
                        "scope": workspace_scope,
                        "context_object": {
                            "metadata": {
                                "memory_kind": "policy_memory",
                                "state": "ACTIVE",
                                "canonical_rule_type": "action_auto_execute",
                                "related_action": "capture",
                                "scope": malformed_scope,
                                "revisions": [{"value_fields": {"canonical_value": "forbidden"}}],
                            }
                        },
                    },
                )
                action_update = self.op(
                    OperationAction.UPDATE,
                    target="policy",
                    payload={
                        "action": "capture",
                        "auto_execute_allowed": True,
                        "scope": workspace_scope,
                    },
                    context_type=ContextType.ACTION_POLICY,
                )

                self.resolver.resolve([policy_memory, action_update])

                self.assertTrue(action_update.payload["auto_execute_allowed"])

    def test_missing_metadata_scope_can_use_strict_top_level_scope(self) -> None:
        workspace_scope = {
            "applicability": {
                "all_of": [{"namespace": "memoryos", "kind": "workspace", "id": "w1"}]
            }
        }
        policy_memory = self.op(
            OperationAction.ADD,
            target="rule",
            payload={
                "memory_type": "project_rule",
                "scope": workspace_scope,
                "context_object": {
                    "metadata": {
                        "memory_kind": "policy_memory",
                        "state": "ACTIVE",
                        "canonical_rule_type": "action_auto_execute",
                        "related_action": "capture",
                        "revisions": [{"value_fields": {"canonical_value": "forbidden"}}],
                    }
                },
            },
        )
        action_update = self.op(
            OperationAction.UPDATE,
            target="policy",
            payload={
                "action": "capture",
                "auto_execute_allowed": True,
                "scope": workspace_scope,
            },
            context_type=ContextType.ACTION_POLICY,
        )

        self.resolver.resolve([policy_memory, action_update])

        self.assertFalse(action_update.payload["auto_execute_allowed"])

    def test_workspace_rule_applies_to_more_specific_principal_policy(self) -> None:
        workspace = {"namespace": "memoryos", "kind": "workspace", "id": "w1"}
        principal = {"namespace": "memoryos", "kind": "principal", "id": "u1"}
        policy_memory = self.op(
            OperationAction.ADD,
            target="rule",
            payload={
                "tenant_id": "t1",
                "memory_type": "project_rule",
                "context_object": {
                    "tenant_id": "t1",
                    "metadata": {
                        "memory_kind": "policy_memory",
                        "state": "ACTIVE",
                        "canonical_rule_type": "action_auto_execute",
                        "related_action": "capture",
                        "scope": {"applicability": {"all_of": [workspace]}},
                        "revisions": [{"value_fields": {"canonical_value": "forbidden"}}],
                    },
                },
            },
        )
        action_update = self.op(
            OperationAction.UPDATE,
            target="policy",
            payload={
                "tenant_id": "t1",
                "action": "capture",
                "auto_execute_allowed": True,
                "scope": {"applicability": {"all_of": [workspace, principal]}},
            },
            context_type=ContextType.ACTION_POLICY,
        )

        self.resolver.resolve([policy_memory, action_update])

        self.assertFalse(action_update.payload["auto_execute_allowed"])

    def test_global_rule_stays_within_user_and_tenant_boundary(self) -> None:
        global_scope = {
            "applicability": {
                "all_of": [{"namespace": "memoryos", "kind": "global", "id": "tenant"}]
            }
        }
        workspace_scope = {
            "applicability": {
                "all_of": [{"namespace": "memoryos", "kind": "workspace", "id": "w1"}]
            }
        }
        policy_memory = self.op(
            OperationAction.ADD,
            target="rule",
            payload={
                "tenant_id": "t1",
                "memory_type": "project_rule",
                "context_object": {
                    "tenant_id": "t1",
                    "metadata": {
                        "memory_kind": "policy_memory",
                        "state": "ACTIVE",
                        "canonical_rule_type": "action_auto_execute",
                        "related_action": "capture",
                        "scope": global_scope,
                        "revisions": [{"value_fields": {"canonical_value": "forbidden"}}],
                    },
                },
            },
        )
        other_tenant = self.op(
            OperationAction.UPDATE,
            target="policy",
            payload={
                "tenant_id": "t2",
                "action": "capture",
                "auto_execute_allowed": True,
                "scope": workspace_scope,
            },
            context_type=ContextType.ACTION_POLICY,
        )
        same_tenant = self.op(
            OperationAction.UPDATE,
            target="same-tenant-policy",
            payload={
                "tenant_id": "t1",
                "action": "capture",
                "auto_execute_allowed": True,
                "scope": workspace_scope,
            },
            context_type=ContextType.ACTION_POLICY,
        )

        self.resolver.resolve([policy_memory, other_tenant, same_tenant])

        self.assertTrue(other_tenant.payload["auto_execute_allowed"])
        self.assertFalse(same_tenant.payload["auto_execute_allowed"])

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
