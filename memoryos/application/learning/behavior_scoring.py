from __future__ import annotations

import math
from datetime import datetime, timezone


class BehaviorEvidenceScorer:
    """Scores behavior evidence using OpenViking-style blended retrieval signals."""

    def __init__(self, recency_half_life_days: float = 14.0) -> None:
        self.recency_half_life_days = recency_half_life_days

    def episode_support(self, weighted_similarity: float, reward: float, created_at: str | None) -> float:
        reward_score = self.reward_score(reward)
        recency_score = self.recency_score(created_at)
        return max(0.0, min(1.0, weighted_similarity * reward_score * recency_score))

    def evidence_confidence(
        self,
        *,
        consistency: float,
        average_support: float,
        sample_count: int,
        distinct_days: int,
        average_reward: float,
    ) -> float:
        sample_strength = self.sample_strength(sample_count)
        diversity_strength = self.diversity_strength(distinct_days)
        reward_score = self.reward_score(average_reward)
        base = (
            consistency * 0.45
            + average_support * 0.25
            + sample_strength * 0.20
            + reward_score * 0.10
        )
        return round(max(0.0, min(1.0, base * diversity_strength)), 6)

    def sample_strength(self, sample_count: int) -> float:
        return round(1.0 - math.exp(-max(sample_count, 0) / 4.0), 6)

    def diversity_strength(self, distinct_days: int) -> float:
        return round(1.0 - math.exp(-max(distinct_days, 0) / 2.0), 6)

    def reward_score(self, reward: float) -> float:
        return max(0.0, min(1.0, (float(reward) + 1.0) / 2.0))

    def recency_score(self, value: str | None) -> float:
        if not value:
            return 0.5
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return 0.5
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        age_days = max((datetime.now(timezone.utc) - parsed).total_seconds() / 86400.0, 0.0)
        decay_rate = math.log(2) / self.recency_half_life_days
        return max(0.0, min(1.0, math.exp(-decay_rate * age_days)))
