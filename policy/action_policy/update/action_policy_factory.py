"""根据已形成的行为证据创建或刷新 ActionPolicy。"""

from __future__ import annotations

from dataclasses import dataclass, field

from policy.action_policy.model.action_policy import ActionPolicy, ActionPolicyStatus
from policy.action_policy.risk import action_spec, canonical_action


@dataclass
class ActionPolicyEvidence:
    user_id: str
    scene_key: str
    action: str
    support_anchor_uri: str
    opportunity_count: int = 0
    activation_count: int = 0
    explicit_authorized: bool = False
    evidence_refs: list[str] = field(default_factory=list)
    supported_behavior_pattern_uris: list[str] = field(default_factory=list)
    constrained_by_support_uris: list[str] = field(default_factory=list)
    required_resource_uris: list[str] = field(default_factory=list)
    required_skill_uris: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not str(self.user_id).strip():
            raise ValueError("ActionPolicyEvidence requires user_id")
        if not str(self.scene_key).strip():
            raise ValueError("ActionPolicyEvidence requires scene_key")
        if not str(self.support_anchor_uri).strip():
            raise ValueError("ActionPolicyEvidence requires support_anchor_uri")
        self.action = canonical_action(self.action)
        if not self.action:
            raise ValueError("ActionPolicyEvidence requires action")
        for field_name in (
            "opportunity_count",
            "activation_count",
        ):
            setattr(self, field_name, max(0, int(getattr(self, field_name))))


class ActionPolicyFactory:
    def build(self, evidence: ActionPolicyEvidence, existing: ActionPolicy | None = None) -> ActionPolicy:
        if existing is not None and (
            existing.user_id != evidence.user_id
            or existing.scene_key != evidence.scene_key
            or existing.action != evidence.action
        ):
            raise ValueError("existing ActionPolicy identity does not match evidence")
        spec = action_spec(evidence.action)
        opportunity_count = max(0, evidence.opportunity_count)
        activation_count = min(opportunity_count, max(0, evidence.activation_count))
        missed_count = max(0, opportunity_count - activation_count)
        activation_rate = activation_count / opportunity_count if opportunity_count else 0.0
        q_value = max(
            0.05,
            min(
                0.95,
                0.25 + activation_rate * 0.55,
            ),
        )
        confidence = max(
            0.1,
            min(
                0.95,
                0.35
                + min(opportunity_count, 8) * 0.06
                + len(evidence.supported_behavior_pattern_uris) * 0.08,
            ),
        )
        auto_execute_allowed = bool(
            spec.risk_level in {"none", "low"}
            and evidence.explicit_authorized
            and spec.executable
            and not spec.requires_confirmation
        )
        if existing is not None:
            can_auto_execute = bool(
                (
                    existing.auto_execute_allowed
                    and existing.status in {ActionPolicyStatus.ACTIVE, ActionPolicyStatus.COOLDOWN}
                    and existing.negative_feedback_count < 3
                )
                or (
                    auto_execute_allowed
                    and existing.status == ActionPolicyStatus.ACTIVE
                    and existing.negative_feedback_count == 0
                )
            )
            return ActionPolicy(
                user_id=existing.user_id,
                scene_key=existing.scene_key,
                action=existing.action,
                support_anchor_uri=existing.support_anchor_uri or evidence.support_anchor_uri,
                policy_id=existing.policy_id,
                # 已存在策略的价值只由显式 reward/penalty 更新，行为快照不能覆盖反馈结果。
                q_value=existing.q_value,
                confidence=max(existing.confidence, confidence),
                reward_score=existing.reward_score,
                penalty_score=existing.penalty_score,
                success_count=existing.success_count,
                failure_count=existing.failure_count,
                opportunity_count=max(existing.opportunity_count, opportunity_count),
                activation_count=max(existing.activation_count, activation_count),
                missed_opportunity_count=max(existing.missed_opportunity_count, missed_count),
                negative_feedback_count=existing.negative_feedback_count,
                status=existing.status,
                auto_execute_allowed=can_auto_execute,
                cooldown_until=existing.cooldown_until,
                evidence_refs=self._merge(existing.evidence_refs, evidence.evidence_refs),
                required_resource_uris=self._merge(existing.required_resource_uris, evidence.required_resource_uris),
                required_skill_uris=self._merge(existing.required_skill_uris, evidence.required_skill_uris),
                supported_behavior_pattern_uris=self._merge(existing.supported_behavior_pattern_uris, evidence.supported_behavior_pattern_uris),
                constrained_by_support_uris=self._merge(
                    existing.constrained_by_support_uris,
                    evidence.constrained_by_support_uris,
                ),
                applied_operation_ids=existing.applied_operation_ids,
                last_rewarded_at=existing.last_rewarded_at,
            )
        return ActionPolicy(
            user_id=evidence.user_id,
            scene_key=evidence.scene_key,
            action=evidence.action,
            support_anchor_uri=evidence.support_anchor_uri,
            q_value=q_value,
            confidence=confidence,
            opportunity_count=opportunity_count,
            activation_count=activation_count,
            missed_opportunity_count=missed_count,
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
