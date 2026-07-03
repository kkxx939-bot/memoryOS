from __future__ import annotations

from memoryos.action_policy.model.action_policy import ActionPolicy, ActionPolicyStatus
from memoryos.core.time import utc_now


class CooldownUpdater:
    def cooldown(self, policy: ActionPolicy, until: str | None = None) -> ActionPolicy:
        policy.status = ActionPolicyStatus.COOLDOWN
        policy.cooldown_until = until
        policy.updated_at = utc_now()
        return policy
