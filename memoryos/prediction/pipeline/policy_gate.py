from __future__ import annotations

from datetime import datetime, timezone

from memoryos.action_policy.model.action_policy import ActionCandidate, ActionPolicy, ActionPolicyStatus
from memoryos.prediction.model.action_context import ActionContext
from memoryos.prediction.model.prediction_result import PolicyDecision
from memoryos.security.action_risk import action_spec


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
        context_block = self._context_block(action_context, action_policy)
        if context_block is not None:
            return context_block
        if action_policy.status in {ActionPolicyStatus.SUPPRESSED, ActionPolicyStatus.DELETED}:
            return PolicyDecision(mode="suppress", allowed=False, action="do_nothing", reason="ActionPolicy is suppressed.")
        if action_policy.status == ActionPolicyStatus.DISABLED_AUTO_EXECUTE:
            return PolicyDecision(mode="ask_user", allowed=True, action="ask_user", reason="Auto execute is disabled.")
        if action_policy.status == ActionPolicyStatus.COOLDOWN:
            return PolicyDecision(mode="ask_user", allowed=True, action="ask_user", reason="ActionPolicy is cooling down.")
        if action_policy.cooldown_until and self._is_future(action_policy.cooldown_until):
            return PolicyDecision(mode="ask_user", allowed=True, action="ask_user", reason="ActionPolicy cooldown is still active.")
        if spec.executable:
            if action_policy.auto_execute_allowed and prediction_confidence >= 0.75:
                return PolicyDecision(mode="execute", allowed=True, action=candidate.action, reason="Low-risk action is authorized.")
            return PolicyDecision(mode="ask_user", allowed=True, action="ask_user", reason="Executable action requires confirmation.")
        if spec.intervenable:
            return PolicyDecision(mode="suggest", allowed=True, action=candidate.action, reason="Candidate is a low-risk suggestion.")
        return PolicyDecision(mode="do_nothing", allowed=True, action="do_nothing", reason="Candidate is not intervenable.")

    def _context_block(self, action_context: ActionContext, action_policy: ActionPolicy) -> PolicyDecision | None:
        memory_text = self._section_text(action_context, "memory_rules").lower()
        if any(token in memory_text for token in ("以后别自动", "不要自动", "禁止自动", "先问我", "no auto", "do not automatically")):
            return PolicyDecision(mode="ask_user", allowed=True, action="ask_user", reason="Policy memory blocks automatic execution.")
        resource_uris = {item.get("uri") for item in self._section_items(action_context, "resource")}
        missing_resources = [uri for uri in action_policy.required_resource_uris if uri not in resource_uris]
        if missing_resources:
            return PolicyDecision(mode="ask_user", allowed=True, action="ask_user", reason="required resource unavailable")
        skill_uris = {item.get("uri") for item in self._section_items(action_context, "skill")}
        missing_skills = [uri for uri in action_policy.required_skill_uris if uri not in skill_uris]
        if missing_skills:
            return PolicyDecision(mode="ask_user", allowed=True, action="ask_user", reason="required skill unavailable")
        recent_text = self._section_text(action_context, "recent_session").lower()
        if any(token in recent_text for token in ("negative_feedback", "explicit_negative", "user_closed", "用户关闭", "负反馈")):
            return PolicyDecision(mode="ask_user", allowed=True, action="ask_user", reason="Recent negative feedback requires confirmation.")
        return None

    def _section_items(self, action_context: ActionContext, section: str) -> list[dict]:
        return list(action_context.packed_context.get("slices", {}).get(section, {}).get("items", []))

    def _section_text(self, action_context: ActionContext, section: str) -> str:
        parts = []
        for item in self._section_items(action_context, section):
            parts.append(str(item.get("content", "")))
            parts.append(str(item.get("metadata", "")))
        return "\n".join(parts)

    def _is_future(self, value: str) -> bool:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return True
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed > datetime.now(timezone.utc)
