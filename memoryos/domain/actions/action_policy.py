from __future__ import annotations

from memoryos.domain.actions.action_schema import canonical_action

INTERVENTION_ACTION_POLICY_VERSION = "intervention_action_policy_v1"

SAFE_INTERVENTIONS = {
    "do_nothing",
    "ask_user",
    "ask_before_turning_on_ac",
    "remind_no_smoking",
    "suggest_break",
}


def preferred_interventions_for(action: str) -> list[str]:
    canonical = canonical_action(action)
    if canonical == "turn_on_ac":
        return ["turn_on_ac", "ask_before_turning_on_ac", "ask_user", "do_nothing"]
    if canonical == "smoke":
        return ["remind_no_smoking", "ask_user", "do_nothing"]
    if canonical == "take_break":
        return ["suggest_break", "ask_user", "do_nothing"]
    if canonical in {"continue_working", "continue_current_activity"}:
        return ["do_nothing", "ask_user"]
    return ["ask_user", "do_nothing"]


def is_safe_intervention(action: str) -> bool:
    return canonical_action(action) in SAFE_INTERVENTIONS or action in SAFE_INTERVENTIONS
