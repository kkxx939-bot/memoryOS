"""通过反馈更新 ActionPolicy 的价值和生命周期。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from foundation.clock import utc_now
from policy.action_policy.model.action_policy import ActionPolicy, ActionPolicyStatus
from policy.action_policy.model.reward_signal import PenaltySignal, RewardSignal


class ActionPolicyUpdater:
    DEFAULT_COOLDOWN = timedelta(hours=24)

    def reward(self, policy: ActionPolicy, signal: RewardSignal, operation_id: str | None = None) -> ActionPolicy:
        self._require_mutable(policy)
        if self._already_applied(policy, operation_id):
            return policy
        delta = max(0.0, min(1.0, float(signal.reward))) * (0.08 if "implicit" in signal.signal_type else 0.16)
        policy.q_value = min(1.0, policy.q_value + delta)
        policy.reward_score += max(0.0, signal.reward)
        policy.success_count += 1
        policy.activation_count += 1
        policy.last_rewarded_at = utc_now()
        policy.updated_at = utc_now()
        if signal.evidence_uri:
            policy.evidence_refs.append(signal.evidence_uri)
        self._mark_applied(policy, operation_id)
        return policy

    def penalize(self, policy: ActionPolicy, signal: PenaltySignal, operation_id: str | None = None) -> ActionPolicy:
        self._require_mutable(policy)
        if self._already_applied(policy, operation_id):
            return policy
        strength = max(0.0, min(1.0, float(signal.penalty)))
        policy.q_value = max(0.0, policy.q_value - strength * 0.18)
        policy.penalty_score += strength
        policy.failure_count += 1
        policy.negative_feedback_count += 1
        policy.updated_at = utc_now()
        if signal.evidence_uri:
            policy.evidence_refs.append(signal.evidence_uri)
        if signal.explicit_rule:
            self._mark_applied(policy, operation_id)
            return self.disable_auto_execute(policy)
        if policy.negative_feedback_count >= 3:
            policy.auto_execute_allowed = False
            policy.status = ActionPolicyStatus.DISABLED_AUTO_EXECUTE
            policy.cooldown_until = None
        else:
            self.cooldown(policy, cooldown_until=None)
        self._mark_applied(policy, operation_id)
        return policy

    def disable_auto_execute(self, policy: ActionPolicy, operation_id: str | None = None) -> ActionPolicy:
        self._require_mutable(policy)
        if self._already_applied(policy, operation_id):
            return policy
        policy.auto_execute_allowed = False
        policy.status = ActionPolicyStatus.DISABLED_AUTO_EXECUTE
        policy.cooldown_until = None
        policy.updated_at = utc_now()
        self._mark_applied(policy, operation_id)
        return policy

    def suppress(self, policy: ActionPolicy, operation_id: str | None = None) -> ActionPolicy:
        self._require_mutable(policy)
        if self._already_applied(policy, operation_id):
            return policy
        policy.status = ActionPolicyStatus.SUPPRESSED
        policy.auto_execute_allowed = False
        policy.updated_at = utc_now()
        self._mark_applied(policy, operation_id)
        return policy

    def cooldown(self, policy: ActionPolicy, cooldown_until: str | None, operation_id: str | None = None) -> ActionPolicy:
        self._require_mutable(policy)
        if self._already_applied(policy, operation_id):
            return policy
        if cooldown_until is None:
            cooldown_until = (datetime.now(timezone.utc) + self.DEFAULT_COOLDOWN).isoformat()
        else:
            self._parse_timestamp(cooldown_until)
        policy.status = ActionPolicyStatus.COOLDOWN
        policy.cooldown_until = cooldown_until
        policy.updated_at = utc_now()
        self._mark_applied(policy, operation_id)
        return policy

    def _already_applied(self, policy: ActionPolicy, operation_id: str | None) -> bool:
        return bool(operation_id and operation_id in policy.applied_operation_ids)

    def _mark_applied(self, policy: ActionPolicy, operation_id: str | None) -> None:
        if not operation_id:
            return
        policy.applied_operation_ids.append(operation_id)
        policy.applied_operation_ids = list(dict.fromkeys(policy.applied_operation_ids))[-500:]

    def _require_mutable(self, policy: ActionPolicy) -> None:
        if policy.status in {ActionPolicyStatus.DELETED, ActionPolicyStatus.OBSOLETE}:
            raise ValueError(f"cannot update {policy.status.value} ActionPolicy")

    def _parse_timestamp(self, value: str) -> datetime:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("cooldown_until must be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
