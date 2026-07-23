"""统一上下文记录支持的类型枚举。"""

from __future__ import annotations

from enum import Enum


class ContextType(str, Enum):
    BEHAVIOR_SUPPORT = "behavior_support"
    ACTION_POLICY_SUPPORT = "action_policy_support"
    BEHAVIOR_CASE = "behavior_case"
    BEHAVIOR_CLUSTER = "behavior_cluster"
    BEHAVIOR_PATTERN = "behavior_pattern"
    ACTION_POLICY = "action_policy"
    SESSION = "session"
    RESOURCE = "resource"
    SKILL = "skill"
