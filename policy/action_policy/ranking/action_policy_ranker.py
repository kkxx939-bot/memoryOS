"""动作策略里的动作策略排序器。"""

from __future__ import annotations

from collections.abc import Collection

from policy.action_policy.risk import action_spec
from policy.action_policy.model.action_policy import ActionCandidate, ActionPolicy, ActionPolicyStatus


class ActionPolicyRanker:
    def rank(
        self,
        policies: list[ActionPolicy],
        similarity_scores: dict[str, float] | None = None,
        context_feasibility: dict[str, float] | None = None,
        verified_support_anchor_uris: Collection[str] | None = None,
    ) -> list[ActionCandidate]:
        similarity_scores = similarity_scores or {}
        context_feasibility = context_feasibility or {}
        verified_anchors = {
            str(uri) for uri in (verified_support_anchor_uris or ()) if str(uri)
        }
        candidates = []
        for policy in policies:
            spec = action_spec(policy.action)
            if not spec.predictable or policy.status in {
                ActionPolicyStatus.SUPPRESSED,
                ActionPolicyStatus.OBSOLETE,
                ActionPolicyStatus.DELETED,
            }:
                continue
            safety_risk = {"none": 0.0, "low": 0.05, "medium": 0.35, "high": 1.0, "private": 1.0}.get(spec.risk_level, 0.8)
            features = {
                "similarity_score": similarity_scores.get(policy.uri, 0.5),
                "behavior_pattern_confidence": policy.confidence,
                "q_value": policy.q_value,
                "reward_score_normalized": min(1.0, policy.reward_score / 10.0),
                "support_anchor_match": 1.0
                if policy.support_anchor_uri and policy.support_anchor_uri in verified_anchors
                else 0.0,
                "context_feasibility": context_feasibility.get(policy.uri, 0.5),
                "penalty_score": min(1.0, policy.penalty_score / 10.0),
                "safety_risk": safety_risk,
                "scene_scope_match": 0.0 if policy.cross_scene_fallback else 1.0,
            }
            score = (
                features["similarity_score"] * 0.25
                + features["behavior_pattern_confidence"] * 0.20
                + features["q_value"] * 0.25
                + features["reward_score_normalized"] * 0.10
                + features["support_anchor_match"] * 0.10
                + features["context_feasibility"] * 0.10
                + features["scene_scope_match"] * 0.15
                - features["penalty_score"] * 0.20
                - features["safety_risk"] * 0.50
            )
            candidates.append(
                ActionCandidate(
                    action=policy.action,
                    score=round(max(0.0, min(1.0, score)), 6),
                    policy_uri=policy.uri,
                    reason="Ranked by ActionPolicy evidence, context, scene scope, and safety.",
                    features=features,
                )
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates
