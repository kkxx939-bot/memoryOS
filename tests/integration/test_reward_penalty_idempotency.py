from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore
from memoryos.contextdb.transaction.recovery import RecoveryService
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class RewardPenaltyIdempotencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = FileSystemSourceStore(self.root)
        self.index = InMemoryIndexStore()
        self.committer = OperationCommitter(self.source, self.index, str(self.root))
        self.policy = ActionPolicy(user_id="u1", scene_key="hot", action="turn_on_ac", memory_anchor_uri="memoryos://user/u1/memories/anchors/hot")
        self.source.write_object(self.policy.to_context_object(), content="policy")
        self.index.upsert_index(self.policy.to_context_object(), content="policy")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def op(self, action: OperationAction, payload: dict) -> ContextOperation:
        return ContextOperation(user_id="u1", context_type=ContextType.ACTION_POLICY, action=action, target_uri=self.policy.uri, payload=payload, operation_id=f"op-{action.value}")

    def test_reward_and_penalty_apply_once_per_operation_id(self) -> None:
        reward = self.op(OperationAction.REWARD, {"reward": 1.0, "signal_type": "explicit_positive"})
        self.committer.commit("u1", [reward])
        first = self.source.read_object(self.policy.uri).metadata
        self.committer.commit("u1", [reward])
        second = self.source.read_object(self.policy.uri).metadata
        self.assertEqual(first["success_count"], second["success_count"])
        self.assertEqual(first["reward_score"], second["reward_score"])
        self.assertEqual(first["q_value"], second["q_value"])
        penalty = self.op(OperationAction.PENALIZE, {"penalty": 1.0})
        self.committer.commit("u1", [penalty])
        third = self.source.read_object(self.policy.uri).metadata
        self.committer.commit("u1", [penalty])
        fourth = self.source.read_object(self.policy.uri).metadata
        self.assertEqual(third["failure_count"], fourth["failure_count"])
        self.assertEqual(third["penalty_score"], fourth["penalty_score"])

    def test_recovery_source_written_does_not_reapply_reward(self) -> None:
        reward = self.op(OperationAction.REWARD, {"reward": 1.0, "signal_type": "explicit_positive"})
        self.committer.commit("u1", [reward])
        first = self.source.read_object(self.policy.uri).metadata
        self.committer.redo.begin(reward, phase="source_written")
        RecoveryService(self.committer.redo, self.committer).recover("u1")
        second = self.source.read_object(self.policy.uri).metadata
        self.assertEqual(first["success_count"], second["success_count"])
        self.assertEqual(first["reward_score"], second["reward_score"])


if __name__ == "__main__":
    unittest.main()
