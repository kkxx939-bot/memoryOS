from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.core.ids import stable_hash
from memoryos.core.time import utc_now
from memoryos.security.action_risk import canonical_action


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
    memory_anchor_uri: str
    policy_id: str = ""
    q_value: float = 0.5
    confidence: float = 0.5
    reward_score: float = 0.0
    penalty_score: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    neutral_count: int = 0
    opportunity_count: int = 0
    activation_count: int = 0
    missed_opportunity_count: int = 0
    negative_feedback_count: int = 0
    status: ActionPolicyStatus = ActionPolicyStatus.ACTIVE
    auto_execute_allowed: bool = False
    cooldown_until: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    required_context_types: list[str] = field(default_factory=lambda: ["memory", "behavior_pattern", "resource", "skill"])
    required_resource_uris: list[str] = field(default_factory=list)
    required_skill_uris: list[str] = field(default_factory=list)
    supported_behavior_pattern_uris: list[str] = field(default_factory=list)
    constrained_by_memory_uris: list[str] = field(default_factory=list)
    applied_operation_ids: list[str] = field(default_factory=list)
    last_opportunity_at: str | None = None
    last_activated_at: str | None = None
    last_rewarded_at: str | None = None
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.memory_anchor_uri:
            raise ValueError("ActionPolicy requires memory_anchor_uri")
        self.action = canonical_action(self.action)
        if not self.policy_id:
            self.policy_id = stable_hash([self.user_id, self.scene_key, self.action], length=16)
        if isinstance(self.status, str):
            self.status = ActionPolicyStatus(self.status)
        self.q_value = max(0.0, min(1.0, float(self.q_value)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.applied_operation_ids = list(dict.fromkeys(self.applied_operation_ids))[-500:]

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
            "neutral_count": self.neutral_count,
            "opportunity_count": self.opportunity_count,
            "activation_count": self.activation_count,
            "missed_opportunity_count": self.missed_opportunity_count,
            "negative_feedback_count": self.negative_feedback_count,
            "status": self.status.value,
            "auto_execute_allowed": self.auto_execute_allowed,
            "cooldown_until": self.cooldown_until,
            "memory_anchor_uri": self.memory_anchor_uri,
            "evidence_refs": self.evidence_refs,
            "required_context_types": self.required_context_types,
            "required_resource_uris": self.required_resource_uris,
            "required_skill_uris": self.required_skill_uris,
            "supported_behavior_pattern_uris": self.supported_behavior_pattern_uris,
            "constrained_by_memory_uris": self.constrained_by_memory_uris,
            "applied_operation_ids": self.applied_operation_ids[-500:],
            "last_opportunity_at": self.last_opportunity_at,
            "last_activated_at": self.last_activated_at,
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
