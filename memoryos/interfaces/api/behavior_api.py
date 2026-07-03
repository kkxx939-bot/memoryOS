from __future__ import annotations

from memoryos.application.learning.behavior_patterns import BehaviorPatternStore
from memoryos.infrastructure.repositories.memory_repository import MemoryStore


def behavior_patterns(store: MemoryStore, payload: dict) -> dict:
    distribution = BehaviorPatternStore(store.root).distribution_for_scene(
        user_id=str(payload["user_id"]),
        retrieval_query=str(payload.get("query", "")),
        context_tags=[str(tag) for tag in payload.get("context_tags", [])],
        limit=int(payload.get("limit", 8)),
    )
    return {"results": distribution}
