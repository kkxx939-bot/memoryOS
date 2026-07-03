from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memoryos.adapters.persistence.sqlite.sqlite_memory_repository import MemoryStore
from memoryos.ports.repositories.memory_repository import MemoryRepository


class RepositoryProtocolTest(unittest.TestCase):
    def test_memory_store_satisfies_memory_repository_protocol_and_exposes_facades(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))

            self.assertIsInstance(store, MemoryRepository)
            self.assertTrue(hasattr(store, "metadata_repository"))
            self.assertTrue(hasattr(store, "search_repository"))
            self.assertTrue(hasattr(store, "lifecycle_repository"))


if __name__ == "__main__":
    unittest.main()
