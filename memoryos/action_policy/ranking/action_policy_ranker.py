"""动作策略里的动作策略排序器。"""

from __future__ import annotations

from memoryos.action_policy.model.action_policy import ActionCandidate, ActionPolicy
from memoryos.security.action_risk import action_spec


class ActionPolicyRanker:
    def rank(
        self,
        policies: list[ActionPolicy],
        similarity_scores: dict[str, float] | None = None,
        context_feasibility: dict[str, float] | None = None,
    ) -> list[ActionCandidate]:
        similarity_scores = similarity_scores or {}
        context_feasibility = context_feasibility or {}
        candidates = []
        for policy in policies:
            spec = action_spec(policy.action)
            safety_risk = {"none": 0.0, "low": 0.05, "medium": 0.35, "high": 1.0, "private": 1.0}.get(spec.risk_level, 0.8)
            features = {
                "similarity_score": similarity_scores.get(policy.uri, 0.5),
                "behavior_pattern_confidence": policy.confidence,
                "q_value": policy.q_value,
                "reward_score_normalized": min(1.0, policy.reward_score / 10.0),
                "memory_anchor_match": 1.0 if policy.memory_anchor_uri else 0.0,
                "context_feasibility": context_feasibility.get(policy.uri, 0.5),
                "penalty_score": min(1.0, policy.penalty_score / 10.0),
                "safety_risk": safety_risk,
            }
            score = (
                features["similarity_score"] * 0.25
                + features["behavior_pattern_confidence"] * 0.20
                + features["q_value"] * 0.25
                + features["reward_score_normalized"] * 0.10
                + features["memory_anchor_match"] * 0.10
                + features["context_feasibility"] * 0.10
                - features["penalty_score"] * 0.20
                - features["safety_risk"] * 0.50
            )
            candidates.append(
                ActionCandidate(
                    action=policy.action,
                    score=round(max(0.0, min(1.0, score)), 6),
                    policy_uri=policy.uri,
                    reason="Ranked by predictive context action policy formula.",
                    features=features,
                )
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates
