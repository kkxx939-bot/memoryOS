from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memoryos.adapters.persistence.sqlite.sqlite_memory_repository import MemoryStore
from memoryos.domain.memory.memory_item import MemoryItem
from memoryos.services.memory.extractor import MemoryOperation
from memoryos.services.memory.memory_operation_validator import MemoryOperationValidator
from memoryos.services.memory.update_service import MemoryUpdateContext, MemoryUpdateService


class MemoryValidationTest(unittest.TestCase):
    def test_sensitive_memory_requires_confirmation(self) -> None:
        operation = MemoryOperation(
            action="add",
            memory_type="event",
            title="secret",
            text="用户的 password 是 123",
            tags=["event"],
        )

        validation = MemoryOperationValidator().validate(operation)

        self.assertFalse(validation.accepted)
        self.assertTrue(validation.sensitive)
        self.assertTrue(validation.needs_user_confirmation)

    def test_delete_diff_includes_tombstone_and_index_delete_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            store.init("gulf")
            item = MemoryItem(user_id="gulf", memory_type="event", title="temp", text="temporary fact", tags=["event"])
            path = store.add_memory(item).relative_to(Path(tmp)).as_posix()
            operation = MemoryOperation(
                action="delete",
                memory_type="event",
                title="delete",
                text="forget temporary fact",
                tags=["event", "user_confirmed"],
                target=path,
            )

            diff = MemoryUpdateService(store).apply(
                [operation],
                MemoryUpdateContext(user_id="gulf", source="test", diff_id="delete-test", explicit_user_intent=True),
            )

            deletion = diff["operations"]["deletes"][0]
            self.assertIn("tombstone", deletion)
            self.assertEqual(deletion["index_delete_job"]["status"], "delete_pending")


if __name__ == "__main__":
    unittest.main()
