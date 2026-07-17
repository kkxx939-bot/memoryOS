from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.connect import ConnectMetadata
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.session.session_commit import SessionCommitService
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore, InMemoryQueueStore
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.prediction.model.prediction_request import PredictionRequest


class SessionCommitRequiresCommitterTest(unittest.TestCase):
    def test_requires_committer_unless_plan_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = SessionArchive(user_id="u1", session_id="s1", archive_uri="memoryos://user/u1/sessions/history/s1", messages=[{"content": "记住我喜欢 26 度"}])
            service = SessionCommitService(SessionArchiveStore(tmp), InMemoryQueueStore())
            with self.assertRaises(RuntimeError):
                service.async_commit(archive)
            plan_store = SessionArchiveStore(tmp)
            plan_service = SessionCommitService(plan_store, InMemoryQueueStore(), allow_plan_only=True)
            plan_service.async_commit(archive)
            self.assertEqual(plan_store.read_async_outputs(archive)["memory_diff"]["status"], "planned")

    def test_committer_and_client_produce_committed_diffs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = FileSystemSourceStore(root)
            index = InMemoryIndexStore()
            committer = OperationCommitter(source, index, str(root))
            store = SessionArchiveStore(root)
            service = SessionCommitService(store, InMemoryQueueStore(), committer=committer)
            archive = SessionArchive(user_id="u1", session_id="s1", archive_uri="memoryos://user/u1/sessions/history/s1", messages=[{"content": "记住我喜欢 26 度"}])
            service.async_commit(archive)
            payload = store.read_async_outputs(archive)["memory_diff"]
            self.assertEqual(payload["status"], "committed")
            client = MemoryOSClient(str(root))
            policy = ActionPolicy(user_id="u1", scene_key="hot", action="turn_on_ac", memory_anchor_uri="memoryos://user/u1/memories/anchors/hot")
            client.source_store.write_object(policy.to_context_object(), content=json.dumps(policy.to_dict()))
            client.index_store.upsert_index(policy.to_context_object(), content="hot turn_on_ac")
            result = client.process_observation(
                PredictionRequest(
                    user_id="u1",
                    episode_id="s2",
                    observation={"scene": "hot", "signals": ["hot_environment"]},
                    available_actions=["turn_on_ac"],
                    connect_metadata=ConnectMetadata.action_capable_embodied("reachy_mini").to_dict(),
                ),
                [policy],
                async_commit=True,
            )
            self.assertEqual(result.prediction_result.memory_operations, [])
            persisted = client.session_archive_store.read_archive(
                "memoryos://user/u1/sessions/history/s2"
            )
            diff = client.session_archive_store.read_async_outputs(persisted)["memory_diff"]
            self.assertEqual(diff["status"], "committed")


if __name__ == "__main__":
    unittest.main()
