from __future__ import annotations

import json

from memoryos.adapters.events.jsonl_outbox import FeedbackEventStore
from memoryos.ports.repositories.memory_repository import MemoryRepository
from memoryos.security.path_safety import validate_identifier
from memoryos.services.learning.learning_service import LearningProcessor


class ReplayWorker:
    def __init__(self, store: MemoryRepository) -> None:
        self.store = store
        self.events = FeedbackEventStore(store.root)
        self.learning = LearningProcessor(store)

    def replay_feedback(self, user_id: str, limit: int | None = None) -> dict:
        validate_identifier(user_id, "user_id")
        events = self.events.feedback_events(user_id, limit=limit)
        results = []
        for event in events:
            episode_id = str(event.get("episode_id", ""))
            episode_result = self._read_episode_result(user_id, episode_id)
            learning_result = self.learning.apply_feedback_event(event, episode_result)
            results.append(
                {
                    "event_id": event.get("event_id"),
                    "episode_id": episode_id,
                    "idempotent": learning_result.get("idempotent", False),
                    "learning_result": learning_result,
                }
            )
        return {
            "user_id": user_id,
            "replayed": len(results),
            "idempotent": sum(1 for item in results if item.get("idempotent")),
            "results": results,
        }

    def _read_episode_result(self, user_id: str, episode_id: str) -> dict:
        path = self.store.root / "user" / user_id / "episodes" / episode_id / "episode_result.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
