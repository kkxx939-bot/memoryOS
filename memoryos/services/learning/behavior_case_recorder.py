from __future__ import annotations

from pathlib import Path

from memoryos.services.learning.behavior_patterns import BehaviorPatternStore


class BehaviorCaseRecorder:
    def __init__(self, root: Path) -> None:
        self.patterns = BehaviorPatternStore(root)

    def record_case(
        self,
        *,
        user_id: str,
        episode_id: str,
        retrieval_query: str,
        context_tags: list[str],
        predicted_action: str,
        actual_action: str,
        reward: float,
        created_at: str,
        predicted_candidates: list[dict],
        action_params: dict,
        scene_features: dict,
        spontaneity: str,
        intervention: str,
        intervention_result: str,
    ) -> dict:
        return self.patterns.record(
            user_id=user_id,
            episode_id=episode_id,
            retrieval_query=retrieval_query,
            context_tags=context_tags,
            predicted_action=predicted_action,
            actual_action=actual_action,
            reward=reward,
            created_at=created_at,
            predicted_candidates=predicted_candidates,
            action_params=action_params,
            scene_features=scene_features,
            spontaneity=spontaneity,
            intervention=intervention,
            intervention_result=intervention_result,
        )
