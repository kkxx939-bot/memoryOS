from __future__ import annotations

from enum import Enum


class ContextType(str, Enum):
    MEMORY = "memory"
    BEHAVIOR_CASE = "behavior_case"
    BEHAVIOR_CLUSTER = "behavior_cluster"
    BEHAVIOR_PATTERN = "behavior_pattern"
    ACTION_POLICY = "action_policy"
    PREDICTION_LEDGER = "prediction_ledger"
    SESSION = "session"
    RESOURCE = "resource"
    SKILL = "skill"
