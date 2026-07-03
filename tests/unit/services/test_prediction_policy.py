from __future__ import annotations

import unittest

from memoryos.services.policy.policy_gate import PermissionPolicyEngine
from memoryos.services.prediction.candidate_generator import Candidate, CandidateGenerator
from memoryos.usecases.intervention.select_intervention import InterventionSelector


class PredictionPolicyTest(unittest.TestCase):
    def test_candidate_carries_action_safety_metadata(self) -> None:
        private = Candidate(action="take_shower", need="comfort", prior=0.9)
        unknown = Candidate(action="unmapped_action", need="unknown", prior=0.8)
        ac = Candidate(action="open_ac", need="cool_down", prior=0.7)

        self.assertTrue(private.is_private)
        self.assertFalse(private.intervenable)
        self.assertEqual(private.risk_level, "private")
        self.assertFalse(unknown.intervenable)
        self.assertEqual(unknown.risk_level, "unknown")
        self.assertEqual(ac.action, "turn_on_ac")
        self.assertTrue(ac.intervenable)

    def test_generator_canonicalizes_memory_actions(self) -> None:
        candidates = CandidateGenerator().generate(
            "hot room",
            memories=[
                {
                    "type": "case",
                    "path": "user/gulf/cases/hot.md",
                    "title": "hot case",
                    "actual_action": "open_ac",
                    "effective_weight": 0.9,
                }
            ],
        )
        actions = {candidate.action for candidate in candidates}

        self.assertIn("turn_on_ac", actions)
        self.assertNotIn("open_ac", actions)

    def test_policy_and_selector_block_private_or_unknown_actions(self) -> None:
        engine = PermissionPolicyEngine()
        self.assertFalse(engine.authorize("take_shower").allowed)
        self.assertFalse(engine.authorize("unmapped_action").allowed)

        selected = InterventionSelector(engine).select(
            Candidate(action="take_shower", need="comfort", prior=0.9, score=0.9),
            ["ask_user", "do_nothing"],
            {},
        )

        self.assertEqual(selected.action, "do_nothing")
        self.assertEqual(selected.features["policy_allowed"], 0.0)


if __name__ == "__main__":
    unittest.main()
