from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from memoryos.action_policy.model.action_policy import ActionCandidate, ActionPolicy
from memoryos.prediction.model.action_context import ActionContext
from memoryos.prediction.pipeline.policy_gate import PolicyGate


class PolicyGateContextRulesTest(unittest.TestCase):
    def context(
        self,
        memory_rules=None,
        resources=None,
        skills=None,
        recent_session=None,
        *,
        anchor: bool = True,
    ) -> ActionContext:
        anchor_items = (
            [
                {
                    "uri": "memoryos://user/u1/memories/anchors/hot",
                    "context_type": "memory",
                    "verified_exact_anchor": True,
                }
            ]
            if anchor
            else []
        )
        return ActionContext(
            user_id="u1",
            candidate_actions=["turn_on_ac"],
            packed_context={
                "slices": {
                    "memory_rules": {"items": memory_rules or []},
                    "memory_anchor": {"items": anchor_items},
                    "resource": {"items": resources or []},
                    "skill": {"items": skills or []},
                    "recent_session": {"items": recent_session or []},
                }
            },
        )

    def policy(self) -> ActionPolicy:
        return ActionPolicy(
            user_id="u1",
            scene_key="hot",
            action="turn_on_ac",
            memory_anchor_uri="memoryos://user/u1/memories/anchors/hot",
            auto_execute_allowed=True,
            required_resource_uris=["memoryos://resources/devices/ac"],
            required_skill_uris=["memoryos://skills/ac-control"],
        )

    def test_context_rules_block_execute(self) -> None:
        candidate = ActionCandidate(action="turn_on_ac", score=0.9, policy_uri="p", reason="test")
        policy = self.policy()
        gate = PolicyGate()
        self.assertEqual(gate.evaluate(candidate, self.context(memory_rules=[{"content": "以后别自动开空调"}], resources=[{"uri": policy.required_resource_uris[0]}], skills=[{"uri": policy.required_skill_uris[0]}]), policy, 0.9).mode, "ask_user")
        self.assertEqual(gate.evaluate(candidate, self.context(skills=[{"uri": policy.required_skill_uris[0]}]), policy, 0.9).reason, "required resource unavailable")
        self.assertEqual(gate.evaluate(candidate, self.context(resources=[{"uri": policy.required_resource_uris[0]}]), policy, 0.9).reason, "required skill unavailable")
        policy.cooldown_until = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        self.assertEqual(gate.evaluate(candidate, self.context(resources=[{"uri": policy.required_resource_uris[0]}], skills=[{"uri": policy.required_skill_uris[0]}]), policy, 0.9).mode, "ask_user")
        policy.cooldown_until = None
        self.assertEqual(gate.evaluate(candidate, self.context(resources=[{"uri": policy.required_resource_uris[0]}], skills=[{"uri": policy.required_skill_uris[0]}], recent_session=[{"content": "negative_feedback"}]), policy, 0.9).mode, "ask_user")
        self.assertEqual(gate.evaluate(candidate, self.context(resources=[{"uri": policy.required_resource_uris[0]}], skills=[{"uri": policy.required_skill_uris[0]}]), policy, 0.9).mode, "execute")

    def test_declared_anchor_missing_or_unverified_blocks_auto_execute(self) -> None:
        policy = self.policy()
        candidate = ActionCandidate(action=policy.action, score=0.9, policy_uri=policy.uri, reason="test")
        missing = PolicyGate().evaluate(
            candidate,
            self.context(
                resources=[{"uri": policy.required_resource_uris[0]}],
                skills=[{"uri": policy.required_skill_uris[0]}],
                anchor=False,
            ),
            policy,
            0.9,
        )
        forged = self.context(
            resources=[{"uri": policy.required_resource_uris[0]}],
            skills=[{"uri": policy.required_skill_uris[0]}],
        )
        forged.packed_context["slices"]["memory_anchor"]["items"][0].pop("verified_exact_anchor")
        unverified = PolicyGate().evaluate(candidate, forged, policy, 0.9)

        self.assertEqual(missing.mode, "ask_user")
        self.assertEqual(unverified.mode, "ask_user")
        self.assertIn("unverified", missing.reason.lower())

    def test_legacy_policy_without_declared_anchor_keeps_existing_gate_semantics(self) -> None:
        policy = self.policy()
        policy.memory_anchor_uri = ""
        candidate = ActionCandidate(action=policy.action, score=0.9, policy_uri=policy.uri, reason="test")

        decision = PolicyGate().evaluate(
            candidate,
            self.context(
                resources=[{"uri": policy.required_resource_uris[0]}],
                skills=[{"uri": policy.required_skill_uris[0]}],
                anchor=False,
            ),
            policy,
            0.9,
        )

        self.assertEqual(decision.mode, "execute")


if __name__ == "__main__":
    unittest.main()
