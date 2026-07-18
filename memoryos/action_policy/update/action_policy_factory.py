"""动作策略里的动作策略工厂。"""

from __future__ import annotations

from dataclasses import dataclass, field

from memoryos.action_policy.model.action_policy import ActionPolicy, ActionPolicyStatus
from memoryos.security.action_risk import action_spec, canonical_action


@dataclass
class ActionPolicyEvidence:
    user_id: str
    scene_key: str
    action: str
    support_anchor_uri: str
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    opportunity_count: int = 0
    activation_count: int = 0
    missed_count: int = 0
    explicit_authorized: bool = False
    evidence_refs: list[str] = field(default_factory=list)
    supported_behavior_pattern_uris: list[str] = field(default_factory=list)
    constrained_by_support_uris: list[str] = field(default_factory=list)
    required_resource_uris: list[str] = field(default_factory=list)
    required_skill_uris: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.action = canonical_action(self.action)


class ActionPolicyFactory:
    def build(self, evidence: ActionPolicyEvidence, existing: ActionPolicy | None = None) -> ActionPolicy:
        spec = action_spec(evidence.action)
        success_count = max(0, evidence.positive_count)
        failure_count = max(0, evidence.negative_count)
        neutral_count = max(0, evidence.neutral_count)
        opportunity_count = max(evidence.opportunity_count, success_count + failure_count + neutral_count)
        activation_count = max(evidence.activation_count, success_count)
        missed_count = max(evidence.missed_count, max(0, opportunity_count - activation_count - failure_count))
        reward_score = round(success_count * 1.0 + activation_count * 0.25, 6)
        penalty_score = round(failure_count * 1.5 + missed_count * 0.25, 6)
        q_value = max(0.05, min(0.95, 0.5 + success_count * 0.08 + activation_count * 0.02 - failure_count * 0.12 - missed_count * 0.03))
        confidence = max(0.1, min(0.95, 0.35 + min(opportunity_count, 8) * 0.06 + len(evidence.supported_behavior_pattern_uris) * 0.08 - failure_count * 0.04))
        auto_execute_allowed = bool(
            spec.risk_level in {"none", "low"}
            and evidence.explicit_authorized
            and failure_count == 0
            and spec.executable
            and not spec.requires_confirmation
        )
        if existing is not None:
            return ActionPolicy(
                user_id=existing.user_id,
                scene_key=existing.scene_key,
                action=existing.action,
                support_anchor_uri=existing.support_anchor_uri or evidence.support_anchor_uri,
                policy_id=existing.policy_id,
                q_value=max(existing.q_value, q_value) if failure_count == 0 else min(existing.q_value, q_value),
                confidence=max(existing.confidence, confidence),
                reward_score=existing.reward_score + reward_score,
                penalty_score=existing.penalty_score + penalty_score,
                success_count=existing.success_count + success_count,
                failure_count=existing.failure_count + failure_count,
                neutral_count=existing.neutral_count + neutral_count,
                opportunity_count=existing.opportunity_count + opportunity_count,
                activation_count=existing.activation_count + activation_count,
                missed_opportunity_count=existing.missed_opportunity_count + missed_count,
                negative_feedback_count=existing.negative_feedback_count + failure_count,
                status=existing.status,
                auto_execute_allowed=existing.auto_execute_allowed or auto_execute_allowed,
                cooldown_until=existing.cooldown_until,
                evidence_refs=self._merge(existing.evidence_refs, evidence.evidence_refs),
                required_context_types=existing.required_context_types,
                required_resource_uris=self._merge(existing.required_resource_uris, evidence.required_resource_uris),
                required_skill_uris=self._merge(existing.required_skill_uris, evidence.required_skill_uris),
                supported_behavior_pattern_uris=self._merge(existing.supported_behavior_pattern_uris, evidence.supported_behavior_pattern_uris),
                constrained_by_support_uris=self._merge(
                    existing.constrained_by_support_uris,
                    evidence.constrained_by_support_uris,
                ),
                applied_operation_ids=existing.applied_operation_ids,
                last_opportunity_at=existing.last_opportunity_at,
                last_activated_at=existing.last_activated_at,
                last_rewarded_at=existing.last_rewarded_at,
            )
        return ActionPolicy(
            user_id=evidence.user_id,
            scene_key=evidence.scene_key,
            action=evidence.action,
            support_anchor_uri=evidence.support_anchor_uri,
            q_value=q_value,
            confidence=confidence,
            reward_score=reward_score,
            penalty_score=penalty_score,
            success_count=success_count,
            failure_count=failure_count,
            neutral_count=neutral_count,
            opportunity_count=opportunity_count,
            activation_count=activation_count,
            missed_opportunity_count=missed_count,
            negative_feedback_count=failure_count,
            status=ActionPolicyStatus.ACTIVE,
            auto_execute_allowed=auto_execute_allowed,
            evidence_refs=evidence.evidence_refs,
            required_resource_uris=evidence.required_resource_uris,
            required_skill_uris=evidence.required_skill_uris,
            supported_behavior_pattern_uris=evidence.supported_behavior_pattern_uris,
            constrained_by_support_uris=evidence.constrained_by_support_uris,
        )

    def _merge(self, left: list[str], right: list[str]) -> list[str]:
        return list(dict.fromkeys([*left, *right]))
