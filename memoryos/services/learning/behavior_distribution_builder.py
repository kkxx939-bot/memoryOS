from __future__ import annotations

from pathlib import Path

from memoryos.services.learning.behavior_patterns import BehaviorPatternStore


class BehaviorDistributionBuilder:
    def __init__(self, root: Path) -> None:
        self.patterns = BehaviorPatternStore(root)

    def distribution_for_scene(
        self,
        *,
        user_id: str,
        retrieval_query: str,
        context_tags: list[str],
        limit: int = 12,
    ) -> list[dict]:
        return self.patterns.distribution_for_scene(
            user_id=user_id,
            retrieval_query=retrieval_query,
            context_tags=context_tags,
            limit=limit,
        )
