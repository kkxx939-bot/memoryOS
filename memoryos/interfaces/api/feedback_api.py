from __future__ import annotations

from memoryos.ports.repositories.memory_repository import MemoryRepository
from memoryos.usecases.feedback.record_feedback import FeedbackService


def record_feedback(store: MemoryRepository, payload: dict) -> dict:
    return FeedbackService(store).record_feedback(
        user_id=str(payload["user_id"]),
        episode_id=str(payload["episode_id"]),
        feedback=str(payload.get("feedback", "")),
        reward=float(payload.get("reward", 0.0)),
        actual_action=payload.get("actual_action"),
        action_params=payload.get("action_params") if isinstance(payload.get("action_params"), dict) else None,
        spontaneity=str(payload.get("spontaneity", "unknown")),
        intervention_result=str(payload.get("intervention_result", "")),
        correction=payload.get("correction"),
        corrects_memory=bool(payload.get("corrects_memory", False)),
    )
