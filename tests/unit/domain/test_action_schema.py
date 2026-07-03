from __future__ import annotations

import unittest

from memoryos.domain.actions.action_policy import is_safe_intervention, preferred_interventions_for
from memoryos.domain.actions.action_schema import action_spec, canonical_action


class ActionSchemaTest(unittest.TestCase):
    def test_unknown_action_is_not_intervenable_by_default(self) -> None:
        spec = action_spec("unknown_robot_action")

        self.assertEqual(spec.action, "unknown_robot_action")
        self.assertEqual(spec.risk_level, "unknown")
        self.assertTrue(spec.predictable)
        self.assertFalse(spec.intervenable)
        self.assertFalse(spec.executable)
        self.assertTrue(spec.requires_confirmation)

    def test_aliases_are_canonicalized_before_policy_lookup(self) -> None:
        self.assertEqual(canonical_action("open_ac"), "turn_on_ac")
        self.assertEqual(preferred_interventions_for("open_ac")[0], "turn_on_ac")
        self.assertTrue(is_safe_intervention("ask_user"))
        self.assertFalse(is_safe_intervention("place_order"))


if __name__ == "__main__":
    unittest.main()
