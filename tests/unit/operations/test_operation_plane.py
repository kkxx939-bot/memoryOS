from __future__ import annotations

import tempfile
import unittest

from memoryos.contextdb.model import ContextObject, ContextType
from memoryos.contextdb.store import FileSystemSourceStore, InMemoryIndexStore
from memoryos.operations.commit import OperationCoalescer, OperationCommitter, RedoLog
from memoryos.operations.model import ContextOperation, OperationAction


class OperationPlaneTest(unittest.TestCase):
    def test_coalesces_add_update_and_add_delete(self) -> None:
        target = "memoryos://user/gulf/memories/anchors/home-comfort"
        add = ContextOperation(
            user_id="gulf",
            context_type=ContextType.MEMORY,
            action=OperationAction.ADD,
            target_uri=target,
            payload={"title": "old"},
        )
        update = ContextOperation(
            user_id="gulf",
            context_type=ContextType.MEMORY,
            action=OperationAction.UPDATE,
            target_uri=target,
            payload={"title": "new"},
        )
        coalesced = OperationCoalescer().coalesce([add, update])
        self.assertEqual(len(coalesced), 1)
        self.assertEqual(coalesced[0].action, OperationAction.ADD)
        self.assertEqual(coalesced[0].payload["title"], "new")

        delete = ContextOperation(
            user_id="gulf",
            context_type=ContextType.MEMORY,
            action=OperationAction.DELETE,
            target_uri=target,
            payload={},
        )
        self.assertEqual(OperationCoalescer().coalesce([add, delete]), [])

    def test_reward_penalty_merge_preserves_both_deltas(self) -> None:
        target = "memoryos://user/gulf/action_policies/hot/turn_on_ac"
        reward = ContextOperation(
            user_id="gulf",
            context_type=ContextType.ACTION_POLICY,
            action=OperationAction.REWARD,
            target_uri=target,
            payload={"reward_delta": 0.4},
        )
        penalty = ContextOperation(
            user_id="gulf",
            context_type=ContextType.ACTION_POLICY,
            action=OperationAction.PENALIZE,
            target_uri=target,
            payload={"penalty_delta": 0.2},
        )
        merged = OperationCoalescer().coalesce([reward, penalty])[0]
        self.assertEqual(merged.payload["reward_delta"], 0.4)
        self.assertEqual(merged.payload["penalty_delta"], 0.2)

    def test_committer_writes_source_index_diff_audit_and_clears_redo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = FileSystemSourceStore(tmp)
            index = InMemoryIndexStore()
            committer = OperationCommitter(source, index, tmp)
            obj = ContextObject(
                uri="memoryos://user/gulf/memories/anchors/home-comfort",
                context_type=ContextType.MEMORY,
                title="Home comfort",
                owner_user_id="gulf",
            )
            op = ContextOperation(
                user_id="gulf",
                context_type=ContextType.MEMORY,
                action=OperationAction.ADD,
                target_uri=obj.uri,
                payload={"context_object": obj.to_dict(), "content": "comfort"},
            )
            diff = committer.commit("gulf", [op])
            self.assertEqual(diff.operations[0].status.value, "committed")
            self.assertEqual(source.read_object(obj.uri).title, "Home comfort")
            self.assertEqual(index.search("comfort", filters={"owner_user_id": "gulf"})[0].uri, obj.uri)
            self.assertEqual(RedoLog(tmp).pending(), [])


if __name__ == "__main__":
    unittest.main()
