from __future__ import annotations

from memoryos.application.episode.episode_service import EpisodeProcessor
from memoryos.infrastructure.repositories.memory_repository import MemoryStore


def process_episode(store: MemoryStore, payload: dict) -> dict:
    return EpisodeProcessor(store).process(
        user_id=str(payload["user_id"]),
        episode_id=str(payload["episode_id"]),
        scene=payload.get("scene"),
        observation=payload.get("observation"),
        messages=payload.get("messages"),
        available_actions=payload.get("available_actions"),
        retrieval_limit=int(payload.get("retrieval_limit", 8)),
        memory_write_timing=payload.get("memory_write_timing"),
        episode_log_timing=str(payload.get("episode_log_timing", "before_prediction")),
        memory_commit_timing=str(payload.get("memory_commit_timing", "after_feedback")),
    )
