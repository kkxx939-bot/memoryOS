"""ActionPolicy 的动作规范、别名和风险目录。"""

from policy.action_policy.risk.catalog import (
    ACTION_SCHEMA_VERSION,
    ACTION_SPECS,
    ActionSpec,
    action_need,
    action_spec,
    canonical_action,
)

__all__ = [
    "ACTION_SCHEMA_VERSION",
    "ACTION_SPECS",
    "ActionSpec",
    "action_need",
    "action_spec",
    "canonical_action",
]
