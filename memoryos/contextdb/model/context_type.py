"""上下文数据库里的上下文类型。"""

from __future__ import annotations

from enum import Enum


class ContextType(str, Enum):
    MEMORY = "memory"
    BEHAVIOR_SUPPORT = "behavior_support"
    ACTION_POLICY_SUPPORT = "action_policy_support"
    BEHAVIOR_CASE = "behavior_case"
    BEHAVIOR_CLUSTER = "behavior_cluster"
    BEHAVIOR_PATTERN = "behavior_pattern"
    ACTION_POLICY = "action_policy"
    PREDICTION_LEDGER = "prediction_ledger"
    SESSION = "session"
    RESOURCE = "resource"
    SKILL = "skill"
