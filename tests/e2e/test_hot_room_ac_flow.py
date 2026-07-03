from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from memoryos.adapters.persistence.sqlite.sqlite_memory_repository import MemoryStore
from memoryos.domain.scene.observation import ObservationContext
from memoryos.services.learning.behavior_patterns import BehaviorPatternStore
from memoryos.usecases.episode.process_observation import EpisodeProcessor


class HotRoomAcFlowTest(unittest.TestCase):
    def test_hot_room_behavior_pattern_predicts_canonical_ac_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = MemoryStore(root)
            pattern_store = BehaviorPatternStore(root)
            observation = ObservationContext(
                raw_text="用户回到房间，说太热并且在出汗。",
                location="room",
                activity="arrive_home",
                signals=["says_hot", "sweating"],
                environment={"temperature": 31, "humidity": 78, "ac_status": "off"},
                observed_at="2026-07-03T20:00:00+08:00",
            )
            for index in range(3):
                pattern_store.record(
                    user_id="gulf",
                    episode_id=f"hot-room-{index}",
                    retrieval_query=observation.to_retrieval_query(),
                    context_tags=observation.context_tags(),
                    predicted_action="seek_cooling",
                    actual_action="open_ac",
                    reward=1.0,
                    created_at=(datetime(2026, 7, 1, tzinfo=timezone.utc) + timedelta(days=index)).isoformat(),
                    action_params={"target_temperature": 26},
                )

            result = EpisodeProcessor(store).process(
                user_id="gulf",
                episode_id="hot-room-current",
                observation=observation,
                available_actions=["turn_on_ac", "ask_user", "do_nothing"],
                memory_write_timing="deferred",
            )

            top = result["ranked_candidates"][0]
            self.assertEqual(top["action"], "turn_on_ac")
            self.assertTrue(top["intervenable"])
            self.assertEqual(top["risk_level"], "low")
            self.assertIn(result["intervention"]["action"], {"ask_user", "turn_on_ac", "do_nothing"})


if __name__ == "__main__":
    unittest.main()
