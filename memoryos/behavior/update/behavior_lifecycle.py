from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from memoryos.behavior.model.behavior_case import BehaviorCase
from memoryos.behavior.model.behavior_pattern import BehaviorCluster, BehaviorPattern
from memoryos.memory.model.memory import MemoryAnchor


@dataclass(frozen=True)
class BehaviorLifecycleResult:
    temporary_cases: list[BehaviorCase]
    cluster: BehaviorCluster | None = None
    pattern: BehaviorPattern | None = None
    memory_anchor: MemoryAnchor | None = None
    memory_candidate_required: bool = False


class BehaviorLifecycleService:
    def evaluate(self, user_id: str, scene_key: str, cases: list[BehaviorCase]) -> BehaviorLifecycleResult:
        relevant = [case for case in cases if case.user_id == user_id and case.scene_key == scene_key]
        if len(relevant) < 2:
            return BehaviorLifecycleResult(temporary_cases=relevant)
        anchor = self._anchor(user_id, scene_key, relevant)
        cluster = BehaviorCluster(
            user_id=user_id,
            scene_key=scene_key,
            memory_anchor_uri=anchor.uri,
            case_refs=[case.case_id for case in relevant],
            confidence=min(0.85, 0.35 + len(relevant) * 0.12),
        )
        if len(relevant) < 3:
            return BehaviorLifecycleResult(temporary_cases=[], cluster=cluster, memory_anchor=anchor)
        pattern = BehaviorPattern(
            user_id=user_id,
            scene_key=scene_key,
            trigger_conditions={"scene_key": scene_key},
            memory_anchor_uri=anchor.uri,
            case_refs=[case.case_id for case in relevant],
            action_distribution=self._action_distribution(relevant),
            hotness=min(1.0, len(relevant) * 0.12),
            confidence=min(0.95, 0.45 + len(relevant) * 0.10),
        )
        return BehaviorLifecycleResult(
            temporary_cases=[],
            cluster=cluster,
            pattern=pattern,
            memory_anchor=anchor,
            memory_candidate_required=True,
        )

    def _anchor(self, user_id: str, scene_key: str, cases: list[BehaviorCase]) -> MemoryAnchor:
        uri = f"memoryos://user/{user_id}/memories/anchors/{scene_key}"
        return MemoryAnchor(
            uri=uri,
            user_id=user_id,
            title=f"Behavior anchor {scene_key}",
            content=(
                "User has a recurring behavior theme in this scene. "
                "Preferences, resources, and automation rules should be observed before durable inference."
            ),
            anchor_key=scene_key,
            confidence=min(0.85, 0.45 + len(cases) * 0.10),
            supporting_behavior_uris=[case.case_id for case in cases],
        )

    def _action_distribution(self, cases: list[BehaviorCase]) -> list[dict]:
        counts = Counter(case.user_actual_action or case.executed_action or case.selected_action or "unknown" for case in cases)
        total = max(1, sum(counts.values()))
        return [
            {"action": action, "count": count, "probability": round(count / total, 6)}
            for action, count in counts.most_common()
        ]
