from __future__ import annotations

from memoryos.behavior.model.observation import Observation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.source_store import IndexStore


class SimilarBehaviorRetriever:
    def __init__(self, index_store: IndexStore) -> None:
        self.index_store = index_store

    def retrieve(self, user_id: str, observation: Observation, limit: int = 8) -> dict:
        query = " ".join([observation.raw_text, observation.location, observation.activity, *observation.signals])
        hits = []
        for context_type in (
            ContextType.BEHAVIOR_PATTERN,
            ContextType.BEHAVIOR_CLUSTER,
            ContextType.BEHAVIOR_CASE,
            ContextType.ACTION_POLICY,
            ContextType.MEMORY,
        ):
            hits.extend(
                self.index_store.search(
                    query or observation.scene_key,
                    filters={"owner_user_id": user_id, "context_type": context_type.value},
                    limit=limit,
                )
            )
        return {
            "query": query,
            "scene_key": observation.scene_key,
            "hits": hits[:limit],
            "similarity_scores": {hit.uri: min(1.0, float(hit.score)) for hit in hits},
        }
