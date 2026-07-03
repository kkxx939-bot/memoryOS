from __future__ import annotations

from memoryos.interfaces.api.request_context import APIRequestContext, user_id_from_context_or_payload
from memoryos.ports.repositories.memory_repository import MemoryRepository
from memoryos.services.learning.behavior_distribution_builder import BehaviorDistributionBuilder


def behavior_patterns(store: MemoryRepository, payload: dict, context: APIRequestContext | None = None) -> dict:
    distribution = BehaviorDistributionBuilder(store.root).distribution_for_scene(
        user_id=user_id_from_context_or_payload(context, payload),
        retrieval_query=str(payload.get("query", "")),
        context_tags=[str(tag) for tag in payload.get("context_tags", [])],
        limit=int(payload.get("limit", 8)),
    )
    return {"results": distribution}
