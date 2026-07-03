from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memoryos.adapters.events.jsonl_outbox import FeedbackEventStore
from memoryos.adapters.persistence.sqlite.sqlite_memory_repository import MemoryStore
from memoryos.workers.feedback_worker import FeedbackWorker


class FailingLearningProcessor:
    def apply_feedback_event(self, event: dict, episode_result: dict) -> dict:
        raise RuntimeError("forced learning failure")


class FeedbackWorkerTest(unittest.TestCase):
    def test_outbox_claim_is_exclusive_until_failure_or_lease_expiry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events = FeedbackEventStore(Path(tmp))
            feedback_event = events.append_feedback_event(
                "gulf",
                "ep_claim",
                {"user_id": "gulf", "episode_id": "ep_claim", "feedback": "ok", "reward": 1},
            )
            events.append_outbox_event("gulf", feedback_event)

            first = events.claim_pending_outbox_events("gulf", worker_id="worker-a", lease_seconds=60)
            second = events.claim_pending_outbox_events("gulf", worker_id="worker-b", lease_seconds=60)

            self.assertEqual(len(first), 1)
            self.assertEqual(first[0]["status"], "processing")
            self.assertEqual(first[0]["locked_by"], "worker-a")
            self.assertEqual(second, [])

    def test_failed_feedback_event_retries_then_dead_letters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            store.init("gulf")
            events = FeedbackEventStore(Path(tmp))
            feedback_event = events.append_feedback_event(
                "gulf",
                "ep_worker",
                {
                    "user_id": "gulf",
                    "episode_id": "ep_worker",
                    "feedback": "bad",
                    "reward": -1,
                },
            )
            events.append_outbox_event("gulf", feedback_event)

            worker = FeedbackWorker(store)
            worker.learning = FailingLearningProcessor()  # type: ignore[assignment]
            first = worker.process_pending("gulf", max_retries=2)
            second = worker.process_pending("gulf", max_retries=2)
            third = worker.process_pending("gulf", max_retries=2)

            self.assertEqual(first["processed"], 0)
            self.assertEqual(first["failed"], 1)
            self.assertEqual(first["failures"][0]["status"], "failed")
            self.assertEqual(second["failures"][0]["status"], "dead_letter")
            self.assertEqual(third["failed"], 0)


if __name__ == "__main__":
    unittest.main()
