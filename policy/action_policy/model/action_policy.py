"""ActionPolicy 的领域模型与在线候选。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from foundation.clock import utc_now
from foundation.ids import stable_hash
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_type import ContextType
from policy.action_policy.risk import action_spec, canonical_action


class ActionPolicyStatus(str, Enum):
    ACTIVE = "active"
    COOLDOWN = "cooldown"
    SUPPRESSED = "suppressed"
    DISABLED_AUTO_EXECUTE = "disabled_auto_execute"
    OBSOLETE = "obsolete"
    DELETED = "deleted"


@dataclass
class ActionPolicy:
    user_id: str
    scene_key: str
    action: str
    support_anchor_uri: str
    policy_id: str = ""
    q_value: float = 0.5
    confidence: float = 0.5
    reward_score: float = 0.0
    penalty_score: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    opportunity_count: int = 0
    activation_count: int = 0
    missed_opportunity_count: int = 0
    negative_feedback_count: int = 0
    status: ActionPolicyStatus = ActionPolicyStatus.ACTIVE
    auto_execute_allowed: bool = False
    cooldown_until: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    required_resource_uris: list[str] = field(default_factory=list)
    required_skill_uris: list[str] = field(default_factory=list)
    supported_behavior_pattern_uris: list[str] = field(default_factory=list)
    constrained_by_support_uris: list[str] = field(default_factory=list)
    # 只描述本次检索是否跨场景，不属于可持久化策略事实。
    cross_scene_fallback: bool = field(default=False, repr=False, compare=False)
    applied_operation_ids: list[str] = field(default_factory=list)
    last_rewarded_at: str | None = None
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not str(self.user_id).strip():
            raise ValueError("ActionPolicy requires user_id")
        if not str(self.scene_key).strip():
            raise ValueError("ActionPolicy requires scene_key")
        if not self.support_anchor_uri:
            raise ValueError("ActionPolicy requires support_anchor_uri")
        self.action = canonical_action(self.action)
        if not self.action:
            raise ValueError("ActionPolicy requires action")
        if not self.policy_id:
            self.policy_id = stable_hash([self.user_id, self.scene_key, self.action], length=16)
        if isinstance(self.status, str):
            self.status = ActionPolicyStatus(self.status)
        self.q_value = self._bounded_score(self.q_value, "q_value")
        self.confidence = self._bounded_score(self.confidence, "confidence")
        self.reward_score = self._nonnegative_score(self.reward_score, "reward_score")
        self.penalty_score = self._nonnegative_score(self.penalty_score, "penalty_score")
        self.auto_execute_allowed = self.auto_execute_allowed is True
        self.cross_scene_fallback = self.cross_scene_fallback is True
        for field_name in (
            "success_count",
            "failure_count",
            "opportunity_count",
            "activation_count",
            "missed_opportunity_count",
            "negative_feedback_count",
        ):
            setattr(self, field_name, max(0, int(getattr(self, field_name))))
        for field_name in (
            "evidence_refs",
            "required_resource_uris",
            "required_skill_uris",
            "supported_behavior_pattern_uris",
            "constrained_by_support_uris",
        ):
            values = getattr(self, field_name) or []
            setattr(self, field_name, list(dict.fromkeys(str(item) for item in values if str(item))))
        self.applied_operation_ids = list(dict.fromkeys(self.applied_operation_ids or []))[-500:]

        # 持久化数据可能来自旧版本或外部注入，领域模型必须再次执行安全收口。
        spec = action_spec(self.action)
        if (
            not spec.executable
            or spec.requires_confirmation
            or spec.risk_level not in {"none", "low"}
            or self.status
            in {
                ActionPolicyStatus.DISABLED_AUTO_EXECUTE,
                ActionPolicyStatus.SUPPRESSED,
                ActionPolicyStatus.OBSOLETE,
                ActionPolicyStatus.DELETED,
            }
        ):
            self.auto_execute_allowed = False

    @staticmethod
    def _bounded_score(value: float, field_name: str) -> float:
        score = float(value)
        if not math.isfinite(score):
            raise ValueError(f"{field_name} must be finite")
        return max(0.0, min(1.0, score))

    @staticmethod
    def _nonnegative_score(value: float, field_name: str) -> float:
        score = float(value)
        if not math.isfinite(score) or score < 0.0:
            raise ValueError(f"{field_name} must be a finite non-negative number")
        return score

    @property
    def uri(self) -> str:
        return f"memoryos://user/{self.user_id}/action_policies/{self.scene_key}/{self.action}"

    def to_context_object(self) -> ContextObject:
        return ContextObject(
            uri=self.uri,
            context_type=ContextType.ACTION_POLICY,
            title=f"ActionPolicy {self.scene_key}/{self.action}",
            owner_user_id=self.user_id,
            hotness=self.confidence,
            behavior_support_hotness=self.q_value,
            metadata=self.to_dict(),
            updated_at=self.updated_at,
        )

    def to_dict(self) -> dict:
        return {
            "policy_id": self.policy_id,
            "user_id": self.user_id,
            "scene_key": self.scene_key,
            "action": self.action,
            "q_value": self.q_value,
            "confidence": self.confidence,
            "reward_score": self.reward_score,
            "penalty_score": self.penalty_score,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "opportunity_count": self.opportunity_count,
            "activation_count": self.activation_count,
            "missed_opportunity_count": self.missed_opportunity_count,
            "negative_feedback_count": self.negative_feedback_count,
            "status": self.status.value,
            "auto_execute_allowed": self.auto_execute_allowed,
            "cooldown_until": self.cooldown_until,
            "support_anchor_uri": self.support_anchor_uri,
            "evidence_refs": self.evidence_refs,
            "required_resource_uris": self.required_resource_uris,
            "required_skill_uris": self.required_skill_uris,
            "supported_behavior_pattern_uris": self.supported_behavior_pattern_uris,
            "constrained_by_support_uris": self.constrained_by_support_uris,
            "applied_operation_ids": self.applied_operation_ids[-500:],
            "last_rewarded_at": self.last_rewarded_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class ActionCandidate:
    action: str
    score: float
    policy_uri: str
    reason: str
    features: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        action = canonical_action(self.action)
        score = float(self.score)
        if not action:
            raise ValueError("ActionCandidate requires action")
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("ActionCandidate score must be finite and between 0 and 1")
        if not str(self.policy_uri).strip():
            raise ValueError("ActionCandidate requires policy_uri")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "score", score)
