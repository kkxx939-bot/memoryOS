from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore, InMemoryLockStore
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class OperationCommitterPathLockTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
