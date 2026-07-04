from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore
from memoryos.contextdb.transaction.recovery import RecoveryService
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class RedoRecoveryTest(unittest.TestCase):
    def test_source_written_recovery_rebuilds_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = FileSystemSourceStore(root)
            index = InMemoryIndexStore()
            committer = OperationCommitter(source, index, str(root))
            obj = ContextObject(uri="memoryos://user/u1/memories/preferences/temp", context_type=ContextType.MEMORY, title="temperature", owner_user_id="u1")
            operation = ContextOperation(
                user_id="u1",
                context_type=ContextType.MEMORY,
                action=OperationAction.ADD,
                target_uri=obj.uri,
                payload={"context_object": obj.to_dict(), "content": "prefers 26 degree"},
            )
            source.write_object(obj, content="prefers 26 degree")
            committer.redo.begin(operation, phase="source_written")
            result = RecoveryService(committer.redo, committer).recover("u1")
            self.assertEqual(result.recovered_count, 1)
            self.assertTrue(index.search("26", filters={"owner_user_id": "u1", "context_type": "memory"}))
            self.assertFalse(committer.redo.pending())


if __name__ == "__main__":
    unittest.main()
