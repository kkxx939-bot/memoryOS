"""动作策略里的动作策略更新器。"""

from __future__ import annotations

from memoryos.action_policy.model.action_policy import ActionPolicy, ActionPolicyStatus
from memoryos.action_policy.model.reward_signal import PenaltySignal, RewardSignal
from memoryos.core.clock import utc_now


class ActionPolicyUpdater:
    def reward(self, policy: ActionPolicy, signal: RewardSignal, operation_id: str | None = None) -> ActionPolicy:
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
        else:
            policy.status = ActionPolicyStatus.COOLDOWN
        self._mark_applied(policy, operation_id)
        return policy

    def disable_auto_execute(self, policy: ActionPolicy, operation_id: str | None = None) -> ActionPolicy:
        if self._already_applied(policy, operation_id):
            return policy
        policy.auto_execute_allowed = False
        policy.status = ActionPolicyStatus.DISABLED_AUTO_EXECUTE
        policy.updated_at = utc_now()
        self._mark_applied(policy, operation_id)
        return policy

    def suppress(self, policy: ActionPolicy, operation_id: str | None = None) -> ActionPolicy:
        if self._already_applied(policy, operation_id):
            return policy
        policy.status = ActionPolicyStatus.SUPPRESSED
        policy.auto_execute_allowed = False
        policy.updated_at = utc_now()
        self._mark_applied(policy, operation_id)
        return policy

    def cooldown(self, policy: ActionPolicy, cooldown_until: str | None, operation_id: str | None = None) -> ActionPolicy:
        if self._already_applied(policy, operation_id):
            return policy
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
