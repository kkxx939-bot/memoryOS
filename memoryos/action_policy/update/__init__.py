"""Stable, lazily resolved ActionPolicy update exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_PUBLIC_ATTRS = {
    "ActionPolicyUpdater": (
        "memoryos.action_policy.update.action_policy_updater",
        "ActionPolicyUpdater",
    ),
    "FeedbackCommitPlanner": (
        "memoryos.application.memory.feedback_commit_planner",
        "FeedbackCommitPlanner",
    ),
}

if TYPE_CHECKING:
    from memoryos.action_policy.update.action_policy_updater import ActionPolicyUpdater
    from memoryos.application.memory.feedback_commit_planner import FeedbackCommitPlanner


def __getattr__(name: str) -> Any:
    target = _PUBLIC_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})

__all__ = ["ActionPolicyUpdater", "FeedbackCommitPlanner"]
