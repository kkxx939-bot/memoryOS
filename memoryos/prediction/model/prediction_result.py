from __future__ import annotations

from dataclasses import dataclass, field

from memoryos.action_policy.model.action_policy import ActionCandidate
from memoryos.behavior.model.observation import Observation
from memoryos.prediction.model.action_context import ActionContext


@dataclass(frozen=True)
class PolicyDecision:
    mode: str
    allowed: bool
    action: str
    reason: str
    policy_version: str = "predictive_policy_gate_v1"

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "allowed": self.allowed,
            "action": self.action,
            "reason": self.reason,
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True)
class PredictionResult:
    request_id: str
    episode_id: str
    observation: Observation
    candidates: list[ActionCandidate]
    action_context: ActionContext
    decision: PolicyDecision
    memory_operations: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.memory_operations:
            raise ValueError("PredictionResult must not contain durable memory operations")

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "observation": self.observation.__dict__,
            "candidates": [candidate.__dict__ for candidate in self.candidates],
            "action_context": self.action_context.to_dict(),
            "decision": self.decision.to_dict(),
            "memory_operations": [],
        }
