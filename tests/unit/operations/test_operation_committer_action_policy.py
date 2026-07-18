from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memoryos.action_policy.integration.commit_registration import (
    register_default_action_policy_commit_handlers,
)
from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class OperationCommitterActionPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        register_default_action_policy_commit_handlers()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = FileSystemSourceStore(self.root)
        self.index = InMemoryIndexStore()
        self.committer = OperationCommitter(self.source, self.index, str(self.root))
        self.policy = ActionPolicy(
            user_id="u1",
            scene_key="hot_room",
            action="turn_on_air_conditioner",
            support_anchor_uri="memoryos://user/u1/support/behavior/home_comfort",
            auto_execute_allowed=True,
        )
        self.source.write_object(self.policy.to_context_object(), content="policy content")
        self.index.upsert_index(
            self.policy.to_context_object(),
            content="policy content",
            tenant_id="default",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def op(self, action: OperationAction, payload: dict) -> ContextOperation:
        return ContextOperation(
            user_id="u1",
            context_type=ContextType.ACTION_POLICY,
            action=action,
            target_uri=self.policy.uri,
            payload=payload,
        )

    def stored(self) -> dict:
        return self.source.read_object(self.policy.uri).metadata

    def test_reward_updates_q_value_reward_score_and_success_count(self) -> None:
        self.committer.commit("u1", [self.op(OperationAction.REWARD, {"reward": 1.0, "signal_type": "explicit_positive"})])
        stored = self.stored()
        self.assertGreater(stored["q_value"], 0.5)
        self.assertEqual(stored["reward_score"], 1.0)
        self.assertEqual(stored["success_count"], 1)

    def test_penalize_updates_penalty_failure_and_negative_count(self) -> None:
        self.committer.commit("u1", [self.op(OperationAction.PENALIZE, {"penalty": 0.5})])
        stored = self.stored()
        self.assertGreater(stored["penalty_score"], 0)
        self.assertEqual(stored["failure_count"], 1)
        self.assertEqual(stored["negative_feedback_count"], 1)

    def test_three_penalties_disable_auto_execute(self) -> None:
        for _ in range(3):
            self.committer.commit("u1", [self.op(OperationAction.PENALIZE, {"penalty": 0.5})])
        stored = self.stored()
        self.assertFalse(stored["auto_execute_allowed"])
        self.assertEqual(stored["status"], "disabled_auto_execute")

    def test_explicit_rule_penalty_disables_auto_execute(self) -> None:
        self.committer.commit("u1", [self.op(OperationAction.PENALIZE, {"penalty": 1.0, "explicit_rule": "以后别自动开空调"})])
        stored = self.stored()
        self.assertFalse(stored["auto_execute_allowed"])
        self.assertEqual(stored["status"], "disabled_auto_execute")

    def test_cooldown_suppress_disable(self) -> None:
        self.committer.commit("u1", [self.op(OperationAction.COOLDOWN, {"cooldown_until": "2026-07-05T00:00:00Z"})])
        self.assertEqual(self.stored()["cooldown_until"], "2026-07-05T00:00:00Z")
        self.assertEqual(self.stored()["status"], "cooldown")
        self.committer.commit("u1", [self.op(OperationAction.SUPPRESS, {})])
        self.assertEqual(self.stored()["status"], "suppressed")
        self.assertFalse(self.stored()["auto_execute_allowed"])
        self.committer.commit("u1", [self.op(OperationAction.DISABLE, {})])
        self.assertEqual(self.stored()["status"], "disabled_auto_execute")

    def test_compress_does_not_delete_source_content(self) -> None:
        ordinary_obj = ContextObject(
            uri="memoryos://user/u1/behavior_cases/compress-me",
            context_type=ContextType.BEHAVIOR_CASE,
            title="compressible behavior case",
            owner_user_id="u1",
        )
        self.source.write_object(ordinary_obj, content="full source text")
        self.index.upsert_index(ordinary_obj, content="full source text", tenant_id="default")
        self.committer.commit(
            "u1",
            [
                ContextOperation(
                    user_id="u1",
                    context_type=ContextType.BEHAVIOR_CASE,
                    action=OperationAction.COMPRESS,
                    target_uri=ordinary_obj.uri,
                    payload={"reason": "cold memory"},
                )
            ],
        )
        self.assertEqual(self.source.read_content(ordinary_obj.uri), "full source text")


if __name__ == "__main__":
    unittest.main()
