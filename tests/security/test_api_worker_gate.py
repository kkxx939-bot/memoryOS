from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memoryos.adapters.persistence.sqlite.sqlite_memory_repository import MemoryStore
from memoryos.config.settings import Settings
from memoryos.interfaces.api.request_context import APIRequestContext
from memoryos.interfaces.api.worker_api import process_feedback_outbox


class APIWorkerGateTest(unittest.TestCase):
    def test_worker_api_requires_internal_token_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            settings = Settings(memory_root=Path(tmp), worker_internal_token="secret")

            with self.assertRaises(PermissionError):
                process_feedback_outbox(store, {"user_id": "gulf"}, settings=settings)

            result = process_feedback_outbox(
                store,
                {"user_id": "ignored"},
                context=APIRequestContext(user_id="gulf", is_internal_worker=True, token="secret"),
                settings=settings,
            )
            self.assertEqual(result["processed"], 0)


if __name__ == "__main__":
    unittest.main()
