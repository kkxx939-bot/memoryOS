from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memoryos.adapters.persistence.sqlite.sqlite_memory_repository import MemoryStore
from memoryos.domain.feedback.reward_result import compute_rewards
from memoryos.services.learning.learning_service import LearningProcessor


class LearningProcessorTest(unittest.TestCase):
    def test_learning_event_is_idempotent_and_case_memory_keeps_reward_breakdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp))
            store.init("gulf")
            reward_breakdown = compute_rewards(
                predicted_action="continue_current_activity",
                actual_action="open_ac",
                user_reward=1,
                intervention_action="ask_user",
                intervention_result="accepted",
            ).to_dict()
            event = {
                "event_id": "feedback_event_1",
                "user_id": "gulf",
                "episode_id": "ep_learning",
                "payload": {
                    "user_id": "gulf",
                    "episode_id": "ep_learning",
                    "feedback": "accepted",
                    "reward": 1,
                    "reward_breakdown": reward_breakdown,
                    "predicted_action": "continue_current_activity",
                    "actual_action": "open_ac",
                    "recommended_intervention": "ask_user",
                    "intervention_result": "accepted",
                },
            }
            episode_result = {
                "scene": "用户回到房间，说热并出汗。",
                "retrieval_query": "用户回到房间，说热并出汗。",
                "context_tags": ["room", "hot", "sweating"],
                "prediction": {
                    "predicted_action": "continue_current_activity",
                    "recommended_intervention": "ask_user",
                },
                "ranked_candidates": [],
            }

            first = LearningProcessor(store).apply_feedback_event(event, episode_result)
            second = LearningProcessor(store).apply_feedback_event(event, episode_result)
            cases = store.hybrid_search("feedback_event_1", user_id="gulf", memory_type="case", limit=8)

            self.assertFalse(first["idempotent"])
            self.assertTrue(second["idempotent"])
            self.assertEqual(len([row for row in cases if "feedback_event:feedback_event_1" in row["tags"]]), 1)
            self.assertIn("reward_breakdown", cases[0]["content"])
            self.assertGreaterEqual(first["case_memory"]["confidence"], 0.9)


if __name__ == "__main__":
    unittest.main()
