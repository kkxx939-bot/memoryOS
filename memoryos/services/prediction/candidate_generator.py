from __future__ import annotations

from dataclasses import dataclass, field

from memoryos.domain.actions.action_schema import action_need, canonical_action


@dataclass
class Candidate:
    action: str
    need: str
    prior: float
    sources: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    score: float = 0.0
    features: dict[str, float] = field(default_factory=dict)
    memory_evidence: list[dict] = field(default_factory=list)
    reason: str = ""
    used_memories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "need": self.need,
            "prior": self.prior,
            "sources": self.sources,
            "evidence": self.evidence,
            "score": self.score,
            "features": self.features,
            "memory_evidence": self.memory_evidence,
            "reason": self.reason,
            "used_memories": self.used_memories,
        }


class CandidateGenerator:
    def __init__(self, min_history_confidence: float = 0.45) -> None:
        self.min_history_confidence = min_history_confidence

    def generate(
        self,
        scene: str,
        memories: list[dict],
        behavior_patterns: list[dict] | None = None,
        behavior_distribution: list[dict] | None = None,
    ) -> list[Candidate]:
        candidates: dict[str, Candidate] = {}
        self._add_baseline(candidates)
        self._add_behavior_pattern_candidates(candidates, behavior_patterns or [])
        self._add_memory_candidates(candidates, memories)
        return sorted(candidates.values(), key=lambda candidate: candidate.prior, reverse=True)

    def _add_behavior_pattern_candidates(
        self,
        candidates: dict[str, Candidate],
        behavior_patterns: list[dict],
    ) -> None:
        for item in behavior_patterns:
            distribution = item.get("action_distribution") or []
            if distribution:
                self._add_behavior_distribution_candidates(candidates, item, distribution)
                continue
            sample_count = int(item.get("sample_count", 0))
            distinct_days = int(item.get("distinct_days", 0))
            action = str(item["action"])
            support_ratio = float(item.get("prior", 0.0))
            evidence_confidence = float(item.get("evidence_confidence", item.get("prediction_coefficient", 0.0)))
            if evidence_confidence < self.min_history_confidence:
                continue
            source = str(item.get("source", "behavior_pattern"))
            prior = min(0.95, 0.35 + support_ratio * 0.25 + evidence_confidence * 0.40)
            if prior <= 0:
                continue
            self._merge_candidate(
                candidates,
                Candidate(
                    action=action,
                    need=self._need_for_action(action),
                    prior=prior,
                    sources=[source],
                    evidence=[
                        (
                            f"behavior pattern from {sample_count} samples across {distinct_days} days supports {action}; "
                            f"confidence={evidence_confidence:.2f}; prior={prior:.2f}"
                        ),
                        *[
                            f"{episode.get('episode_id')} -> {episode.get('actual_action')}"
                            for episode in item.get("episodes", [])[:3]
                        ],
                    ],
                    reason="Aggregated behavior pattern supports this action as the user's likely outcome.",
                ),
            )

    def _add_behavior_distribution_candidates(
        self,
        candidates: dict[str, Candidate],
        pattern_item: dict,
        distribution: list[dict],
    ) -> None:
        match_strength = float(pattern_item.get("similarity", 1.0)) * float(pattern_item.get("match_weight", 1.0))
        distinct_days = int(pattern_item.get("distinct_days", 0))
        for action_item in distribution:
            action = str(action_item.get("action", ""))
            if not action:
                continue
            probability = float(action_item.get("probability", action_item.get("ratio", 0.0)) or 0.0)
            avg_reward = float(action_item.get("avg_reward", action_item.get("average_reward", 0.0)) or 0.0)
            confidence = float(action_item.get("confidence", action_item.get("evidence_confidence", 0.0)) or 0.0)
            recency_weight = float(action_item.get("recency_weight", 0.5) or 0.0)
            if confidence < self.min_history_confidence and distinct_days < 2:
                continue
            if confidence < self.min_history_confidence and probability < 0.10:
                continue
            reward_score = max(0.0, min(1.0, (avg_reward + 1.0) / 2.0))
            negative_penalty = min(0.30, int(action_item.get("negative_count", 0)) * 0.05)
            prior = (
                0.20
                + probability * 0.30
                + reward_score * 0.20
                + confidence * 0.20
                + recency_weight * 0.10
            )
            prior = max(0.0, min(0.95, prior * max(match_strength, 0.2) - negative_penalty))
            self._merge_candidate(
                candidates,
                Candidate(
                    action=action,
                    need=self._need_for_action(action),
                    prior=prior,
                    sources=["behavior_pattern", "behavior_pattern_distribution"],
                    evidence=[
                        (
                            f"behavior group {pattern_item.get('group_id', '')} supports {action}; "
                            f"probability={probability:.2f}; reward={avg_reward:.2f}; confidence={confidence:.2f}"
                        )
                    ],
                    reason="Similar behavior pattern distribution supports this candidate.",
                ),
            )

    def _add_baseline(self, candidates: dict[str, Candidate]) -> None:
        self._merge_candidate(
            candidates,
            Candidate(
                action="continue_current_activity",
                need="none",
                prior=0.25,
                sources=["baseline"],
                evidence=["No stronger candidate may be present."],
                reason="Baseline candidate.",
            ),
        )

    def _add_memory_candidates(
        self,
        candidates: dict[str, Candidate],
        memories: list[dict],
    ) -> None:
        for memory in memories:
            memory_type = str(memory.get("type", ""))
            weight = float(memory.get("effective_weight", memory.get("confidence", 0.5)) or 0.5)
            prior = min(0.85, 0.35 + weight * 0.45)
            inferred_action = self._action_from_memory(memory)
            if not inferred_action:
                continue
            self._merge_candidate(
                candidates,
                Candidate(
                    action=inferred_action,
                    need=self._need_from_memory(memory) or self._need_for_action(inferred_action),
                    prior=prior,
                    sources=[f"memory_{memory_type}"],
                    evidence=[f"{memory.get('path')}: {memory.get('title')}"],
                    reason=f"Retrieved structured memory explicitly supports action {inferred_action}.",
                    used_memories=[memory["path"]],
                ),
            )

    def _merge_candidate(self, candidates: dict[str, Candidate], candidate: Candidate) -> None:
        existing = candidates.get(candidate.action)
        if existing is None:
            candidates[candidate.action] = candidate
            return
        existing.prior = max(existing.prior, candidate.prior)
        existing.sources = sorted(set(existing.sources + candidate.sources))
        existing.evidence.extend(item for item in candidate.evidence if item not in existing.evidence)
        existing.used_memories = sorted(set(existing.used_memories + candidate.used_memories))
        if candidate.prior >= existing.prior:
            existing.reason = candidate.reason

    def _need_for_action(self, action: str) -> str:
        mapping = {
            "continue_current_activity": "none",
        }
        return mapping.get(action, action_need(action))

    def _action_from_memory(self, memory: dict) -> str:
        for key in ("action", "actual_action", "predicted_action"):
            value = str(memory.get(key, "")).strip()
            if value:
                return canonical_action(value)
        tags = [str(tag) for tag in memory.get("tags", [])]
        for tag in tags:
            if tag.startswith("action:"):
                return canonical_action(tag.split(":", 1)[1])
            if tag.startswith("actual_action:"):
                return canonical_action(tag.split(":", 1)[1])
        for line in str(memory.get("content", "")).splitlines():
            lowered = line.lower().strip()
            if lowered.startswith("actual action:"):
                return canonical_action(line.split(":", 1)[1].strip())
        return ""

    def _need_from_memory(self, memory: dict) -> str:
        value = str(memory.get("need", "")).strip()
        if value:
            return value
        for tag in [str(tag) for tag in memory.get("tags", [])]:
            if tag.startswith("need:"):
                return tag.split(":", 1)[1]
        for line in str(memory.get("content", "")).splitlines():
            lowered = line.lower().strip()
            if lowered.startswith("need:"):
                return line.split(":", 1)[1].strip()
        return ""
