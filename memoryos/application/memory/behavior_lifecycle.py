"""行为模块里的行为生命周期。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from memoryos.behavior.model.behavior_case import BehaviorCase
from memoryos.behavior.model.behavior_pattern import BehaviorCluster, BehaviorPattern
from memoryos.behavior.update.behavior_window import BehaviorWindowEvaluator
from memoryos.support import SupportAnchor


@dataclass(frozen=True)
class BehaviorLifecycleResult:
    temporary_cases: list[BehaviorCase]
    cluster: BehaviorCluster | None = None
    pattern: BehaviorPattern | None = None
    support_anchor: SupportAnchor | None = None
    support_candidate_required: bool = False


class BehaviorLifecycleService:
    """负责 BehaviorLifecycleService 这部分逻辑。"""

    def evaluate(self, user_id: str, scene_key: str, cases: list[BehaviorCase]) -> BehaviorLifecycleResult:
        relevant = [case for case in cases if case.user_id == user_id and case.scene_key == scene_key]
        decision = BehaviorWindowEvaluator().evaluate(scene_key, relevant, [])
        if not decision.create_cluster:
            return BehaviorLifecycleResult(temporary_cases=relevant)
        anchor = self._anchor(user_id, scene_key, relevant)
        cluster = self._cluster(user_id, scene_key, anchor.uri, decision.similar_refs_3d)
        if not decision.create_pattern:
            return BehaviorLifecycleResult(temporary_cases=[], cluster=cluster, support_anchor=anchor)
        pattern = self._pattern(
            user_id,
            scene_key,
            anchor.uri,
            decision.similar_refs_30d,
            decision.similarity_key,
            relevant,
        )
        return BehaviorLifecycleResult(
            temporary_cases=[],
            cluster=cluster,
            pattern=pattern,
            support_anchor=anchor,
            support_candidate_required=True,
        )

    def _cluster(self, user_id: str, scene_key: str, anchor_uri: str, case_refs: list[str]) -> BehaviorCluster:
        return BehaviorCluster(
            user_id=user_id,
            scene_key=scene_key,
            support_anchor_uri=anchor_uri,
            case_refs=case_refs,
            confidence=min(0.85, 0.35 + len(case_refs) * 0.12),
        )

    def _pattern(
        self,
        user_id: str,
        scene_key: str,
        anchor_uri: str,
        case_refs: list[str],
        similarity_key: tuple[str, ...],
        cases: list[BehaviorCase],
    ) -> BehaviorPattern:
        return BehaviorPattern(
            user_id=user_id,
            scene_key=scene_key,
            trigger_conditions={"scene_key": scene_key, "context_tags": list(similarity_key)},
            support_anchor_uri=anchor_uri,
            case_refs=case_refs,
            action_distribution=self._action_distribution(cases),
            hotness=min(1.0, len(case_refs) * 0.12),
            confidence=min(0.95, 0.45 + len(case_refs) * 0.10),
        )

    def _anchor(self, user_id: str, scene_key: str, cases: list[BehaviorCase]) -> SupportAnchor:
        uri = f"memoryos://user/{user_id}/support/behavior/{scene_key}"
        return SupportAnchor(
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
