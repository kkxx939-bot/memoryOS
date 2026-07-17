"""Register ActionPolicy-owned commit components with the operation plane."""

from __future__ import annotations

from memoryos.action_policy.integration.commit_handler import ActionPolicyCommitHandler
from memoryos.action_policy.update.action_policy_updater import ActionPolicyUpdater
from memoryos.operations.commit.domain_registry import (
    RegisteredActionPolicyCommitHandlers,
    register_action_policy_commit_handlers,
)


def build_action_policy_commit_handlers() -> RegisteredActionPolicyCommitHandlers:
    return RegisteredActionPolicyCommitHandlers(
        handler=ActionPolicyCommitHandler,
        updater_factory=ActionPolicyUpdater,
    )


def register_default_action_policy_commit_handlers() -> RegisteredActionPolicyCommitHandlers:
    handlers = build_action_policy_commit_handlers()
    register_action_policy_commit_handlers(handlers)
    return handlers


__all__ = [
    "build_action_policy_commit_handlers",
    "register_default_action_policy_commit_handlers",
]
