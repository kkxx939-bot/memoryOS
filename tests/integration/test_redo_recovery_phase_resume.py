from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore
from memoryos.contextdb.transaction.recovery import RecoveryService
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class RedoRecoveryPhaseResumeTest(unittest.TestCase):
    def test_source_written_add_resumes_index_audit_and_diff_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = FileSystemSourceStore(root)
            index = InMemoryIndexStore()
            committer = OperationCommitter(source, index, str(root))
            obj = ContextObject(uri="memoryos://user/u1/memories/preferences/temp", context_type=ContextType.MEMORY, title="temperature", owner_user_id="u1")
            op = ContextOperation(user_id="u1", context_type=ContextType.MEMORY, action=OperationAction.ADD, target_uri=obj.uri, payload={"context_object": obj.to_dict(), "content": "prefers 26"})
            source.write_object(obj, content="prefers 26")
            committer.redo.begin(op, phase="source_written")
            RecoveryService(committer.redo, committer).recover("u1")
            RecoveryService(committer.redo, committer).recover("u1")
            self.assertTrue(index.search("26", filters={"owner_user_id": "u1", "context_type": "memory"}))
            audit_path = root / "system" / "audit" / "u1.jsonl"
            self.assertEqual(len(audit_path.read_text(encoding="utf-8").splitlines()), 1)
            diff_files = list((root / "system" / "diffs").glob(f"diff_{op.operation_id}.json"))
            self.assertEqual(len(diff_files), 1)

    def test_source_written_reward_and_penalty_do_not_reapply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = FileSystemSourceStore(root)
            index = InMemoryIndexStore()
            committer = OperationCommitter(source, index, str(root))
            policy = ActionPolicy(user_id="u1", scene_key="hot", action="turn_on_ac", memory_anchor_uri="memoryos://user/u1/memories/anchors/hot")
            source.write_object(policy.to_context_object(), content=json.dumps(policy.to_dict()))
            index.upsert_index(policy.to_context_object(), content="policy")
            reward = ContextOperation(user_id="u1", context_type=ContextType.ACTION_POLICY, action=OperationAction.REWARD, target_uri=policy.uri, payload={"reward": 1.0}, operation_id="reward-once")
            committer.commit("u1", [reward])
            first = source.read_object(policy.uri).metadata
            committer.redo.begin(reward, phase="source_written")
            RecoveryService(committer.redo, committer).recover("u1")
            second = source.read_object(policy.uri).metadata
            self.assertEqual(first["success_count"], second["success_count"])
            penalty = ContextOperation(user_id="u1", context_type=ContextType.ACTION_POLICY, action=OperationAction.PENALIZE, target_uri=policy.uri, payload={"penalty": 1.0}, operation_id="penalty-once")
            committer.commit("u1", [penalty])
            third = source.read_object(policy.uri).metadata
            committer.redo.begin(penalty, phase="source_written")
            RecoveryService(committer.redo, committer).recover("u1")
            fourth = source.read_object(policy.uri).metadata
            self.assertEqual(third["failure_count"], fourth["failure_count"])


if __name__ == "__main__":
    unittest.main()
