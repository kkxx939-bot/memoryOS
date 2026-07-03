from __future__ import annotations

from memoryos.ports.repositories.memory_repository import MemoryRepository
from memoryos.workers.feedback_worker import FeedbackWorker
from memoryos.workers.memory_consolidation_worker import MemoryConsolidationWorker
from memoryos.workers.reindex_worker import ReindexWorker
from memoryos.workers.replay_worker import ReplayWorker


def process_feedback_outbox(store: MemoryRepository, payload: dict | None = None) -> dict:
    payload = payload or {}
    limit = payload.get("limit")
    max_retries = int(payload.get("max_retries", 3))
    return FeedbackWorker(store).process_pending(
        user_id=payload.get("user_id"),
        limit=int(limit) if limit is not None else None,
        max_retries=max_retries,
    )


def run_reindex(store: MemoryRepository, payload: dict | None = None) -> dict:
    payload = payload or {}
    return ReindexWorker(store).reindex(user_id=payload.get("user_id"))


def run_replay(store: MemoryRepository, payload: dict | None = None) -> dict:
    payload = payload or {}
    return ReplayWorker(store).replay_feedback(
        user_id=str(payload["user_id"]),
        limit=int(payload["limit"]) if payload.get("limit") is not None else None,
    )


def run_memory_consolidation(store: MemoryRepository, payload: dict | None = None) -> dict:
    payload = payload or {}
    allowed_types = payload.get("allowed_types")
    return MemoryConsolidationWorker(store).archive_cold(
        user_id=str(payload["user_id"]),
        limit=int(payload.get("limit", 20)),
        max_hotness=float(payload.get("max_hotness", 0.12)),
        allowed_types={str(item) for item in allowed_types} if isinstance(allowed_types, list) else None,
    )
