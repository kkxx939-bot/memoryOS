from __future__ import annotations

from memoryos.application.feedback.feedback_service import FeedbackService
from memoryos.infrastructure.repositories.memory_repository import MemoryStore
from memoryos.workers.feedback_worker import FeedbackWorker


def record_feedback(store: MemoryStore, payload: dict) -> dict:
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


def process_feedback_outbox(store: MemoryStore, payload: dict | None = None) -> dict:
    payload = payload or {}
    limit = payload.get("limit")
    return FeedbackWorker(store).process_pending(
        user_id=payload.get("user_id"),
        limit=int(limit) if limit is not None else None,
    )
