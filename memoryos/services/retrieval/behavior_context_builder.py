from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from memoryos.ports.providers.rerank_provider import RerankProvider
from memoryos.services.learning.behavior_feedback import BehaviorStats
from memoryos.services.learning.behavior_patterns import BehaviorPatternStore


@dataclass
class BehaviorRoute:
    retrieval_type: str
    strategy: str
    selected_count: int
    reason: str
    target_uri: str = ""
    level: str = ""
    score: float = 0.0
    match_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "retrieval_type": self.retrieval_type,
            "strategy": self.strategy,
            "selected_count": self.selected_count,
            "reason": self.reason,
            "target_uri": self.target_uri,
            "level": self.level,
            "score": self.score,
            "match_reason": self.match_reason,
        }


@dataclass
class BehaviorContext:
    behavior_patterns: list[dict] = field(default_factory=list)
    behavior_distribution: list[dict] = field(default_factory=list)
    route_trace: list[BehaviorRoute] = field(default_factory=list)

    def source_summary(self) -> dict[str, dict]:
        summary = {}
        if self.behavior_patterns:
            summary["behavior_pattern"] = {
                "count": len(self.behavior_patterns),
                "paths": [item.get("pattern_uri") for item in self.behavior_patterns],
            }
        if self.behavior_distribution:
            summary["behavior_feedback"] = {
                "count": len(self.behavior_distribution),
                "paths": [item.get("signature") for item in self.behavior_distribution],
            }
        return summary


class BehaviorContextBuilder:
    def __init__(
        self,
        root: Path,
        behavior_stats_path: Path,
        rerank_provider: RerankProvider | None = None,
    ) -> None:
        self.root = root
        self.behavior_stats_path = behavior_stats_path
        self.rerank_provider = rerank_provider

    def build(self, user_id: str, query: str, context_tags: list[str]) -> BehaviorContext:
        behavior_patterns = BehaviorPatternStore(self.root, rerank_provider=self.rerank_provider).distribution_for_scene(
            user_id=user_id,
            retrieval_query=query,
            context_tags=context_tags,
        )
        behavior_distribution = BehaviorStats(self.behavior_stats_path).distribution_for_scene(
            query,
            context_tags,
        )
        routes = [
            self._pattern_route(user_id, behavior_patterns),
            self._feedback_route(user_id, behavior_distribution),
        ]
        return BehaviorContext(
            behavior_patterns=behavior_patterns,
            behavior_distribution=behavior_distribution,
            route_trace=routes,
        )

    def _pattern_route(self, user_id: str, patterns: list[dict]) -> BehaviorRoute:
        top = patterns[0] if patterns else {}
        score = float(top.get("prediction_coefficient", 0.0) or 0.0)
        return BehaviorRoute(
            retrieval_type="behavior_pattern",
            strategy="hierarchical_behavior_pattern",
            selected_count=len(patterns),
            reason="retrieve aggregated behavior patterns before detailed episodes",
            target_uri=f"memory://user/{user_id}/behavior",
            level=str(top.get("match_level") or "pattern"),
            score=round(score, 6),
            match_reason=(
                f"top_action={top.get('action', '')}; domain={top.get('domain', '')}; "
                f"confidence={float(top.get('evidence_confidence', 0.0) or 0.0):.3f}"
            ),
        )

    def _feedback_route(self, user_id: str, distribution: list[dict]) -> BehaviorRoute:
        top = distribution[0] if distribution else {}
        score = float(top.get("weighted_prior", top.get("prior", 0.0)) or 0.0)
        return BehaviorRoute(
            retrieval_type="behavior_feedback",
            strategy="hierarchical_behavior_signature",
            selected_count=len(distribution),
            reason="retrieve prediction correctness statistics for similar observations",
            target_uri=f"memory://user/{user_id}/behavior_stats",
            level=str(top.get("match_level") or "signature"),
            score=round(score, 6),
            match_reason=(
                f"top_action={top.get('action', '')}; match_weight={float(top.get('match_weight', 0.0) or 0.0):.3f}; "
                f"reward={float(top.get('behavior_reward_score', 0.0) or 0.0):.3f}"
            ),
        )
