from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore
from memoryos.contextdb.store.source_store import IndexHit
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus
from memoryos.operations.resolver.target_resolver import TargetResolver


class OperationCommitterTargetResolverTest(unittest.TestCase):
    def test_update_without_target_resolves_before_apply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = FileSystemSourceStore(root)
            index = InMemoryIndexStore()
            old = ContextObject(uri="memoryos://user/u1/memories/preferences/temp", context_type=ContextType.MEMORY, title="temperature", owner_user_id="u1")
            source.write_object(old, content="old")
            index.upsert_index(old, content="prefers 26 degree")
            updated = ContextObject(uri=old.uri, context_type=ContextType.MEMORY, title="temperature updated", owner_user_id="u1")
            op = ContextOperation(user_id="u1", context_type=ContextType.MEMORY, action=OperationAction.UPDATE, payload={"query": "26 degree", "context_object": updated.to_dict(), "content": "new"})
            diff = OperationCommitter(source, index, str(root)).commit("u1", [op])
            self.assertEqual(diff.operations[0].target_uri, old.uri)
            self.assertEqual(source.read_content(old.uri), "new")

    def test_delete_action_policy_without_target_resolves_by_scene_and_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = FileSystemSourceStore(root)
            index = InMemoryIndexStore()
            uri = "memoryos://user/u1/action_policies/hot_room/turn_on_ac"
            obj = ContextObject(uri=uri, context_type=ContextType.ACTION_POLICY, title="hot_room turn_on_ac", owner_user_id="u1")
            source.write_object(obj, content="policy")
            index.upsert_index(obj, content="hot_room turn_on_ac")
            op = ContextOperation(user_id="u1", context_type=ContextType.ACTION_POLICY, action=OperationAction.DELETE, payload={"scene_key": "hot_room", "action": "turn_on_ac"})
            diff = OperationCommitter(source, index, str(root)).commit("u1", [op])
            self.assertEqual(diff.operations[0].target_uri, uri)
            self.assertEqual(source.read_object(uri).lifecycle_state, LifecycleState.DELETED)
            self.assertFalse(index.search("turn_on_ac", filters={"owner_user_id": "u1", "context_type": "action_policy"}))

    def test_unresolved_supersede_goes_pending_without_apply(self) -> None:
        class LowIndex(InMemoryIndexStore):
            def search(self, query: str, filters: dict | None = None, limit: int = 10) -> list[IndexHit]:
                return [IndexHit(uri="memoryos://user/u1/memories/profile/a", score=0.2, context_type="memory", title="a")]

        with tempfile.TemporaryDirectory() as tmp:
            source = FileSystemSourceStore(tmp)
            index = LowIndex()
            replacement = ContextObject(uri="memoryos://user/u1/memories/profile/new", context_type=ContextType.MEMORY, title="new", owner_user_id="u1")
            op = ContextOperation(user_id="u1", context_type=ContextType.MEMORY, action=OperationAction.SUPERSEDE, payload={"query": "ambiguous", "context_object": replacement.to_dict(), "content": "new"})
            diff = OperationCommitter(source, index, tmp, target_resolver=TargetResolver(index)).commit("u1", [op])
            self.assertFalse(diff.operations)
            self.assertEqual(diff.pending_operations[0].status, OperationStatus.PENDING)
            with self.assertRaises(FileNotFoundError):
                source.read_object(replacement.uri)

    def test_explicit_cross_user_delete_stays_rejected_and_has_no_source_side_effect(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = FileSystemSourceStore(tmp)
            index = InMemoryIndexStore()
            uri = "memoryos://user/u2/memories/profile/occupation"
            target = ContextObject(
                uri=uri,
                context_type=ContextType.MEMORY,
                title="u2 occupation",
                owner_user_id="u2",
            )
            source.write_object(target, content="engineer")
            index.upsert_index(target, content="engineer")
            operation = ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.DELETE,
                target_uri=uri,
                payload={"tenant_id": "default", "memory_type": "profile"},
            )

            diff = OperationCommitter(source, index, tmp).commit("u1", [operation])

            self.assertEqual(diff.operations, [])
            self.assertEqual(diff.pending_operations, [])
            self.assertEqual([item.operation_id for item in diff.rejected_operations], [operation.operation_id])
            self.assertEqual(diff.rejected_operations[0].status, OperationStatus.REJECTED)
            self.assertEqual(source.read_object(uri).lifecycle_state, LifecycleState.ACTIVE)
            self.assertEqual(source.read_content(uri), "engineer")


if __name__ == "__main__":
    unittest.main()
