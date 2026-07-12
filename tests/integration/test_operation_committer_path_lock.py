from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore, InMemoryLockStore
from memoryos.contextdb.store.source_store import LockLostError
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class OperationCommitterPathLockTest(unittest.TestCase):
    def test_in_memory_fenced_section_renews_same_fence_after_ttl_elapses(self) -> None:
        store = InMemoryLockStore()
        token = store.acquire("long-section", ttl_seconds=1)

        with store.fenced((token,), ttl_seconds=1):
            current = store.locks[token.lock_key]
            store.locks[token.lock_key] = (
                current[0],
                current[1],
                datetime.now(timezone.utc) - timedelta(seconds=1),
            )

        store.assert_owned(token)

    def test_same_target_write_is_lock_protected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_store = InMemoryLockStore()
            source = FileSystemSourceStore(root)
            index = InMemoryIndexStore()
            committer = OperationCommitter(source, index, str(root), lock_store=lock_store)
            obj = ContextObject(uri="memoryos://user/u1/memories/preferences/temp", context_type=ContextType.MEMORY, title="temperature", owner_user_id="u1")
            operation = ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.ADD,
                target_uri=obj.uri,
                payload={"context_object": obj.to_dict(), "content": "prefers 26 degree"},
            )
            token = lock_store.acquire(obj.uri)
            try:
                with self.assertRaises(TimeoutError):
                    committer.commit("u1", [operation])
            finally:
                lock_store.release(token)
            committer.commit("u1", [operation])
            self.assertTrue(index.search("26", filters={"owner_user_id": "u1", "context_type": "memory"}))

    def test_lost_fence_stops_before_index_audit_diff_and_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_store = InMemoryLockStore()
            source = FileSystemSourceStore(root)
            index = InMemoryIndexStore()
            committer = OperationCommitter(source, index, str(root), lock_store=lock_store)
            obj = ContextObject(
                uri="memoryos://user/u1/memories/preferences/fenced",
                context_type=ContextType.MEMORY,
                title="fenced",
                owner_user_id="u1",
            )
            operation = ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.ADD,
                target_uri=obj.uri,
                payload={"context_object": obj.to_dict(), "content": "fenced write"},
            )
            original_advance = committer.redo.advance

            def advance_and_steal(operation_arg, phase, **kwargs):  # noqa: ANN001, ANN202
                path = original_advance(operation_arg, phase, **kwargs)
                if phase == "source_written":
                    with lock_store._guard:
                        current = lock_store.locks[obj.uri]
                        stolen_fence = current[1] + 1
                        lock_store.fences[obj.uri] = stolen_fence
                        lock_store.locks[obj.uri] = (
                            "new-owner",
                            stolen_fence,
                            datetime.now(timezone.utc) + timedelta(seconds=30),
                        )
                return path

            committer.redo.advance = advance_and_steal  # type: ignore[assignment]

            with self.assertRaises(LockLostError):
                committer.commit("u1", [operation])

            self.assertNotIn(obj.uri, index.indexed_uris())
            self.assertFalse(list((root / "system" / "audit").glob("*.jsonl")))
            self.assertFalse(list((root / "system" / "diffs").glob("*.json")))
            self.assertFalse((root / "system" / "operations" / f"{operation.operation_id}.json").exists())
            pending = committer.redo.pending_entries()
            self.assertEqual([(entry.operation_id, entry.phase) for entry in pending], [(operation.operation_id, "source_written")])

    def test_regular_batch_holds_all_target_locks_through_diff_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock_store = InMemoryLockStore()
            source = FileSystemSourceStore(root)
            index = InMemoryIndexStore()
            committer = OperationCommitter(source, index, str(root), lock_store=lock_store)
            objects = [
                ContextObject(
                    uri=f"memoryos://user/u1/memories/preferences/{name}",
                    context_type=ContextType.MEMORY,
                    title=name,
                    owner_user_id="u1",
                )
                for name in ("first", "second")
            ]
            operations = [
                ContextOperation(
                    user_id="u1",
                    context_type=ContextType.MEMORY,
                    action=OperationAction.ADD,
                    target_uri=obj.uri,
                    payload={"context_object": obj.to_dict(), "content": obj.title},
                )
                for obj in objects
            ]
            original_write = committer.diff_writer.write

            def assert_locks_then_write(diff):  # noqa: ANN001, ANN202
                self.assertTrue({obj.uri for obj in objects}.issubset(lock_store.locks))
                return original_write(diff)

            committer.diff_writer.write = assert_locks_then_write  # type: ignore[method-assign]

            committer.commit("u1", operations)

            self.assertEqual(lock_store.locks, {})


if __name__ == "__main__":
    unittest.main()
