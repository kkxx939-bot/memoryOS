from __future__ import annotations

from memoryos.interfaces.api.request_context import APIRequestContext, user_id_from_context_or_payload
from memoryos.ports.repositories.memory_repository import MemoryRepository
from memoryos.usecases.episode.process_observation import EpisodeProcessor


def process_episode(store: MemoryRepository, payload: dict, context: APIRequestContext | None = None) -> dict:
    return EpisodeProcessor(store).process(
        user_id=user_id_from_context_or_payload(context, payload),
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
