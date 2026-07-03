from __future__ import annotations

import unittest

from memoryos.services.prediction.candidate_generator import Candidate
from memoryos.usecases.intervention.select_intervention import InterventionSelector


class ActionSafetyTest(unittest.TestCase):
    def test_unknown_action_is_never_suggested_or_executed(self) -> None:
        decision = InterventionSelector().select(
            Candidate(action="unknown_robot_action", need="unknown", prior=0.8, score=0.8),
            ["ask_user", "unknown_robot_action", "do_nothing"],
            {},
        )

        self.assertEqual(decision.action, "do_nothing")
        self.assertEqual(decision.features["policy_allowed"], 0.0)
        self.assertEqual(decision.features["candidate_risk_level"], "unknown")

    def test_private_behavior_is_prediction_only(self) -> None:
        candidate = Candidate(action="take_shower", need="comfort", prior=0.9, score=0.9)

        self.assertTrue(candidate.predictable)
        self.assertFalse(candidate.intervenable)
        self.assertFalse(candidate.executable)


if __name__ == "__main__":
    unittest.main()
