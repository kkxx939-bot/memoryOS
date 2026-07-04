from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.session.session_commit import SessionCommitService
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.local_stores import (
    FileSystemSourceStore,
    InMemoryIndexStore,
    InMemoryQueueStore,
    InMemoryRelationStore,
)
from memoryos.operations.commit.operation_committer import OperationCommitter


class SessionCommitGeneratesDiffsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = SessionArchiveStore(self.root)
        self.queue = InMemoryQueueStore()
        self.source = FileSystemSourceStore(self.root)
        self.index = InMemoryIndexStore()
        self.relations = InMemoryRelationStore()
        self.committer = OperationCommitter(self.source, self.index, str(self.root), relation_store=self.relations)
        self.service = SessionCommitService(self.store, self.queue, committer=self.committer)
        self.archive_uri = "memoryos://user/u1/sessions/history/archive_001"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_async_commit_generates_real_diffs(self) -> None:
        policy = ActionPolicy(
            user_id="u1",
            scene_key="hot_room",
            action="turn_on_ac",
            memory_anchor_uri="memoryos://user/u1/memories/anchors/hot_room_anchor",
        )
        self.source.write_object(policy.to_context_object(), content=json.dumps(policy.to_dict()))
        self.index.upsert_index(policy.to_context_object(), content="hot_room turn_on_ac")
        archive = SessionArchive(
            user_id="u1",
            session_id="s1",
            archive_uri=self.archive_uri,
            messages=[{"role": "user", "content": "记住我不喜欢空调直吹"}],
            observations=[
                {"episode_id": "e1", "scene_key": "hot_room", "indoor_temperature": 30},
                {"episode_id": "e2", "scene_key": "hot_room", "indoor_temperature": 31},
                {"episode_id": "e3", "scene_key": "hot_room", "indoor_temperature": 30},
            ],
            predictions=[
                {
                    "observation": {"scene_key": "hot_room"},
                    "candidates": [{"action": "turn_on_ac", "policy_uri": "memoryos://user/u1/action_policies/hot_room/turn_on_ac"}],
                    "decision": {"action": "turn_on_ac"},
                }
            ],
            feedback=[
                {"episode_id": "e1", "policy_uri": "memoryos://user/u1/action_policies/hot_room/turn_on_ac", "reward": 0.5, "feedback_type": "implicit_positive"},
                {
                    "episode_id": "e3",
                    "policy_uri": "memoryos://user/u1/action_policies/hot_room/turn_on_ac",
                    "reward": -1.0,
                    "feedback_type": "explicit_negative",
                    "explicit_rule": "以后别自动开空调",
                },
            ],
        )
        self.store.write_sync_archive(archive)
        self.service.async_commit(archive)
        directory = ContextURI.parse(self.archive_uri).to_source_path(self.root)
        memory_diff = json.loads((directory / "memory_diff.json").read_text(encoding="utf-8"))
        behavior_diff = json.loads((directory / "behavior_diff.json").read_text(encoding="utf-8"))
        action_policy_diff = json.loads((directory / "action_policy_diff.json").read_text(encoding="utf-8"))
        self.assertEqual(memory_diff["status"], "committed")
        self.assertEqual(behavior_diff["status"], "committed")
        self.assertEqual(action_policy_diff["status"], "committed")
        self.assertTrue(any(op["context_type"] == "memory" and op["action"] == "add" for op in memory_diff["operations"]))
        self.assertTrue(any(op["context_type"] == "behavior_case" and op["action"] == "add" for op in behavior_diff["operations"]))
        self.assertTrue(any(op["context_type"] == "behavior_cluster" and op["action"] == "add" for op in behavior_diff["operations"]))
        self.assertTrue(any(op["context_type"] == "behavior_pattern" and op["action"] == "add" for op in behavior_diff["operations"]))
        pattern_ops = [op for op in behavior_diff["operations"] if op["context_type"] == "behavior_pattern"]
        self.assertTrue(pattern_ops[0]["payload"]["context_object"]["metadata"]["memory_anchor_uri"])
        self.assertTrue(any(op["action"] == "reward" for op in action_policy_diff["operations"]))
        self.assertTrue(any(op["action"] == "penalize" for op in action_policy_diff["operations"]))
        self.assertTrue(any(op["context_type"] == "memory" and op["action"] == "add" for op in action_policy_diff["operations"]))
        self.assertTrue(any(op["context_type"] == "action_policy" and op["action"] == "disable" for op in action_policy_diff["operations"]))


if __name__ == "__main__":
    unittest.main()
