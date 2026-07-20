"""ActionPolicy 在线决策结果。"""

from __future__ import annotations

from dataclasses import dataclass

from behavior.core.model.observation import Observation
from policy.action_policy.decision.action_context import ActionContext
from policy.action_policy.model.action_policy import ActionCandidate


@dataclass(frozen=True)
class PolicyDecision:
    mode: str
    allowed: bool
    action: str
    reason: str
    policy_version: str = "action_policy_gate_v1"

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

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "episode_id": self.episode_id,
            "observation": self.observation.__dict__,
            "candidates": [candidate.__dict__ for candidate in self.candidates],
            "action_context": self.action_context.to_dict(),
            "decision": self.decision.to_dict(),
        }
