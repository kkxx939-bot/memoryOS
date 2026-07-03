from __future__ import annotations

from memoryos.action_policy.model.action_policy import ActionPolicy


class CandidateGenerator:
    def generate(self, policies: list[ActionPolicy]) -> list[ActionPolicy]:
        return [policy for policy in policies if policy.status.value not in {"deleted", "obsolete"}]
