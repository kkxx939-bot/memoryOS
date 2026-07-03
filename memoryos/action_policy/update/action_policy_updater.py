from __future__ import annotations

from memoryos.action_policy.model.action_policy import ActionPolicy, ActionPolicyStatus
from memoryos.action_policy.model.reward_signal import PenaltySignal, RewardSignal
from memoryos.core.time import utc_now


class ActionPolicyUpdater:
    def reward(self, policy: ActionPolicy, signal: RewardSignal) -> ActionPolicy:
        delta = max(0.0, min(1.0, float(signal.reward))) * (0.08 if "implicit" in signal.signal_type else 0.16)
        policy.q_value = min(1.0, policy.q_value + delta)
        policy.reward_score += max(0.0, signal.reward)
        policy.success_count += 1
        policy.activation_count += 1
        policy.last_rewarded_at = utc_now()
        policy.updated_at = utc_now()
        if signal.evidence_uri:
            policy.evidence_refs.append(signal.evidence_uri)
        return policy

    def penalize(self, policy: ActionPolicy, signal: PenaltySignal) -> ActionPolicy:
        strength = max(0.0, min(1.0, float(signal.penalty)))
        policy.q_value = max(0.0, policy.q_value - strength * 0.18)
        policy.penalty_score += strength
        policy.failure_count += 1
        policy.negative_feedback_count += 1
        policy.updated_at = utc_now()
        if signal.evidence_uri:
            policy.evidence_refs.append(signal.evidence_uri)
        if signal.explicit_rule:
            return self.disable_auto_execute(policy)
        if policy.negative_feedback_count >= 3:
            policy.auto_execute_allowed = False
            policy.status = ActionPolicyStatus.DISABLED_AUTO_EXECUTE
        else:
            policy.status = ActionPolicyStatus.COOLDOWN
        return policy

    def disable_auto_execute(self, policy: ActionPolicy) -> ActionPolicy:
        policy.auto_execute_allowed = False
        policy.status = ActionPolicyStatus.DISABLED_AUTO_EXECUTE
        policy.updated_at = utc_now()
        return policy

    def suppress(self, policy: ActionPolicy) -> ActionPolicy:
        policy.status = ActionPolicyStatus.SUPPRESSED
        policy.auto_execute_allowed = False
        policy.updated_at = utc_now()
        return policy
