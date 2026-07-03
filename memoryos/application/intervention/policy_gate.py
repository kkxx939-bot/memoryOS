from __future__ import annotations

from dataclasses import dataclass

from memoryos.domain.actions.action_schema import action_spec, canonical_action

POLICY_VERSION = "policy_v1"


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    mode: str
    reason: str
    max_allowed_action: str
    policy_version: str = POLICY_VERSION

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "mode": self.mode,
            "reason": self.reason,
            "max_allowed_action": self.max_allowed_action,
            "policy_version": self.policy_version,
        }


class PermissionPolicyEngine:
    def authorize(
        self,
        action: str,
        action_params: dict | None = None,
        context: dict | None = None,
        prediction_confidence: float = 0.0,
        policy_stats: dict | None = None,
    ) -> PolicyDecision:
        canonical = canonical_action(action)
        spec = action_spec(canonical)
        policy_stats = policy_stats or {}

        if spec.risk_level in {"high", "private"}:
            return PolicyDecision(
                allowed=False,
                mode="blocked",
                reason=f"Action {canonical} has risk level {spec.risk_level}.",
                max_allowed_action="do_nothing",
            )
        if not spec.intervenable:
            return PolicyDecision(
                allowed=False,
                mode="blocked",
                reason=f"Action {canonical} is predictable but not intervenable.",
                max_allowed_action="do_nothing",
            )
        if spec.executable and self._allowed_without_confirmation(canonical, policy_stats):
            return PolicyDecision(
                allowed=True,
                mode="execute",
                reason=f"Action {canonical} is explicitly allowed without confirmation.",
                max_allowed_action=canonical,
            )
        if spec.executable or spec.requires_confirmation:
            return PolicyDecision(
                allowed=True,
                mode="ask_user",
                reason=f"Action {canonical} requires confirmation before execution.",
                max_allowed_action="ask_user",
            )
        return PolicyDecision(
            allowed=True,
            mode="suggest",
            reason=f"Action {canonical} can be handled as a low-risk suggestion or reminder.",
            max_allowed_action="suggest_or_remind",
        )

    def _allowed_without_confirmation(self, action: str, policy_stats: dict) -> bool:
        for key in (f"permission::{action}", f"permission::{canonical_action(action)}"):
            entry = policy_stats.get(key, {})
            if isinstance(entry, dict) and entry.get("allowed_without_confirmation") is True:
                return True
        return False
