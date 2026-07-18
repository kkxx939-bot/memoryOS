from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.contextdb.session.session_model import SessionArchive


class SessionCommitGeneratesDiffsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.client = MemoryOSClient(str(self.root))
        self.store = self.client.session_archive_store
        self.source = self.client.source_store
        self.index = self.client.index_store
        self.service = self.client.session_commit_service
        self.archive_uri = "memoryos://user/u1/sessions/history/archive_001"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_async_commit_generates_real_diffs(self) -> None:
        policy = ActionPolicy(
            user_id="u1",
            scene_key="hot_room",
            action="turn_on_ac",
            support_anchor_uri="memoryos://user/u1/support/behavior/hot_room_anchor",
        )
        self.source.write_object(policy.to_context_object(), content=json.dumps(policy.to_dict()))
        self.index.upsert_index(
            policy.to_context_object(),
            content="hot_room turn_on_ac",
            tenant_id="default",
        )
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
        persisted = self.store.read_archive(self.archive_uri)
        outputs = self.store.read_async_outputs(persisted)
        memory_diff = outputs["memory_diff"]
        behavior_diff = outputs["behavior_diff"]
        action_policy_diff = outputs["action_policy_diff"]
        self.assertEqual(memory_diff["status"], "committed")
        self.assertEqual(behavior_diff["status"], "committed")
        self.assertEqual(action_policy_diff["status"], "committed")
        self.assertEqual(memory_diff["edit_proposal_count"], 0)
        self.assertEqual(memory_diff["edit_proposal_ids"], [])
        self.assertGreaterEqual(memory_diff["memory_document_change_count"], 1)
        self.assertTrue(memory_diff["effects"])
        self.assertNotIn("operations", memory_diff)
        self.assertGreater(behavior_diff["operation_count"], 0)
        self.assertGreater(action_policy_diff["operation_count"], 0)
        self.assertTrue(behavior_diff["operation_ids"])
        self.assertTrue(action_policy_diff["operation_ids"])
        preference = self.client.memory_document_store.read_raw(
            "default",
            "u1",
            relative_path="preferences.md",
        )
        self.assertIn("不喜欢空调直吹".encode(), preference)


if __name__ == "__main__":
    unittest.main()
