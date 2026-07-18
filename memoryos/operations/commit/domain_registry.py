"""Explicit ActionPolicy composition for the generic operation plane.

Markdown memory has its own document committer and is intentionally absent
from this registry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RegisteredActionPolicyCommitHandlers:
    handler: Any
    updater_factory: Callable[[], Any]


_ACTION_POLICY_HANDLERS: RegisteredActionPolicyCommitHandlers | None = None


def register_action_policy_commit_handlers(
    handlers: RegisteredActionPolicyCommitHandlers,
) -> None:
    global _ACTION_POLICY_HANDLERS
    _ACTION_POLICY_HANDLERS = handlers


def action_policy_commit_handlers() -> RegisteredActionPolicyCommitHandlers | None:
    return _ACTION_POLICY_HANDLERS


__all__ = [
    "RegisteredActionPolicyCommitHandlers",
    "action_policy_commit_handlers",
    "register_action_policy_commit_handlers",
]
