from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memoryos.adapters.events.jsonl_index_jobs import JsonlIndexJobRepository
from memoryos.ports.providers.embedding_provider import HashingEmbeddingProvider
from memoryos.services.indexing.chunking_service import ChunkingService
from memoryos.services.indexing.index_consistency_service import IndexConsistencyService, IndexState
from memoryos.services.indexing.index_job_service import IndexJobService


class IndexingContractsTest(unittest.TestCase):
    def test_chunking_and_index_job_capture_model_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            chunk = ChunkingService().chunks_for_memory(
                {
                    "id": "memory_hot_room_ac",
                    "type": "case",
                    "title": "hot room",
                    "content": "Scene: hot room\nActual action: open_ac\nReward: 1",
                }
            )[0]
            repo = JsonlIndexJobRepository(Path(tmp))
            job = IndexJobService(repo, HashingEmbeddingProvider(dimensions=8)).enqueue_upsert("gulf", chunk)

            self.assertEqual(job["embedding_provider"], "local_hashing")
            self.assertEqual(job["embedding_dimension"], 8)
            self.assertEqual(job["status"], "pending")
            self.assertIn("vector", job["metadata"]["embedding"])

    def test_index_consistency_detects_model_or_hash_drift(self) -> None:
        current = IndexState("a", "hash1", "p", "m1", 8, "sqlite", index_status="indexed")
        desired = IndexState("a", "hash2", "p", "m1", 8, "sqlite", index_status="indexed")

        self.assertTrue(IndexConsistencyService().is_stale(current, desired))
        self.assertFalse(IndexConsistencyService().is_stale(desired, desired))


if __name__ == "__main__":
    unittest.main()
