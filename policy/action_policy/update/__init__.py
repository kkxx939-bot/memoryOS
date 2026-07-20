"""ActionPolicy 更新能力的惰性公开接口。"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_PUBLIC_ATTRS = {
    "ActionPolicyUpdater": (
        "policy.action_policy.update.action_policy_updater",
        "ActionPolicyUpdater",
    ),
    "FeedbackCommitPlanner": (
        "policy.action_policy.update.feedback_commit_planner",
        "FeedbackCommitPlanner",
    ),
    "PolicySupportWriter": (
        "policy.action_policy.update.policy_support_writer",
        "PolicySupportWriter",
    ),
}

if TYPE_CHECKING:
    from policy.action_policy.update.action_policy_updater import ActionPolicyUpdater
    from policy.action_policy.update.feedback_commit_planner import FeedbackCommitPlanner
    from policy.action_policy.update.policy_support_writer import PolicySupportWriter


def __getattr__(name: str) -> Any:
    target = _PUBLIC_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})

__all__ = ["ActionPolicyUpdater", "FeedbackCommitPlanner", "PolicySupportWriter"]
