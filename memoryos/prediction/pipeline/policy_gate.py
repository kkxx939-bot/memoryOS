from __future__ import annotations

from memoryos.action_policy.model.action_policy import ActionCandidate, ActionPolicy, ActionPolicyStatus
from memoryos.domain.actions.action_schema import action_spec
from memoryos.prediction.model.action_context import ActionContext
from memoryos.prediction.model.prediction_result import PolicyDecision


class PolicyGate:
    def evaluate(
        self,
        candidate: ActionCandidate | None,
        action_context: ActionContext,
        action_policy: ActionPolicy | None,
        prediction_confidence: float = 0.0,
    ) -> PolicyDecision:
        if candidate is None or action_policy is None:
            return PolicyDecision(mode="do_nothing", allowed=True, action="do_nothing", reason="No safe candidate.")
        spec = action_spec(candidate.action)
        if spec.risk_level in {"high", "private", "unknown"}:
            return PolicyDecision(mode="blocked", allowed=False, action="do_nothing", reason="Action risk blocks execution.")
        if action_policy.status in {ActionPolicyStatus.SUPPRESSED, ActionPolicyStatus.DELETED}:
            return PolicyDecision(mode="suppress", allowed=False, action="do_nothing", reason="ActionPolicy is suppressed.")
        if action_policy.status == ActionPolicyStatus.DISABLED_AUTO_EXECUTE:
            return PolicyDecision(mode="ask_user", allowed=True, action="ask_user", reason="Auto execute is disabled.")
        if action_policy.status == ActionPolicyStatus.COOLDOWN:
            return PolicyDecision(mode="ask_user", allowed=True, action="ask_user", reason="ActionPolicy is cooling down.")
        if spec.executable:
            if action_policy.auto_execute_allowed and prediction_confidence >= 0.75:
                return PolicyDecision(mode="execute", allowed=True, action=candidate.action, reason="Low-risk action is authorized.")
            return PolicyDecision(mode="ask_user", allowed=True, action="ask_user", reason="Executable action requires confirmation.")
        if spec.intervenable:
            return PolicyDecision(mode="suggest", allowed=True, action=candidate.action, reason="Candidate is a low-risk suggestion.")
        return PolicyDecision(mode="do_nothing", allowed=True, action="do_nothing", reason="Candidate is not intervenable.")
