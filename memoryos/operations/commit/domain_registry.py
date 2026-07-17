"""Operations-owned registry for domain-specific commit components.

The operation plane owns only these narrow registration contracts. Memory and
ActionPolicy provide implementations, while runtime is the explicit
composition root. Importing a domain package never mutates this registry.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


class MemoryCommitHandlerRegistration(Protocol):
    canonical_handler: Any
    canonical_coordinator: Any
    canonical_planning: Any
    final_state_validator_factory: Callable[[Any, Any, Any], Any]
    planning_envelope_store_factory: Callable[[Any, str], Any]
    session_evidence_reader_factory: Callable[[Any, str], Any]
    domain_classifier_binder: Callable[..., Any]
    current_head_integrity_error: type[Exception]
    load_current_head: Callable[..., Any]
    publish_current_head_sets: Callable[..., Any]
    read_committed_canonical: Callable[..., Any]
    committed_content: Callable[..., str]
    committed_relations: Callable[..., Any]
    materialized_current_revision_payload: Callable[..., dict[str, Any]]
    memory_scope_from_dict: Callable[..., Any]
    scope_key_from_payload: Callable[..., str]
    scope_keys_from_payloads: Callable[..., Any]
    relation_domain_policy_factory: Callable[[], Any]


class ActionPolicyCommitHandlerRegistration(Protocol):
    handler: Any
    updater_factory: Callable[[], Any]


@dataclass(frozen=True)
class RegisteredMemoryCommitHandlers:
    canonical_handler: Any
    canonical_coordinator: Any
    canonical_planning: Any
    final_state_validator_factory: Callable[[Any, Any, Any], Any]
    planning_envelope_store_factory: Callable[[Any, str], Any]
    session_evidence_reader_factory: Callable[[Any, str], Any]
    domain_classifier_binder: Callable[..., Any]
    current_head_integrity_error: type[Exception]
    load_current_head: Callable[..., Any]
    publish_current_head_sets: Callable[..., Any]
    read_committed_canonical: Callable[..., Any]
    committed_content: Callable[..., str]
    committed_relations: Callable[..., Any]
    materialized_current_revision_payload: Callable[..., dict[str, Any]]
    memory_scope_from_dict: Callable[..., Any]
    scope_key_from_payload: Callable[..., str]
    scope_keys_from_payloads: Callable[..., Any]
    relation_domain_policy_factory: Callable[[], Any]


@dataclass(frozen=True)
class RegisteredActionPolicyCommitHandlers:
    handler: Any
    updater_factory: Callable[[], Any]


_memory_handlers: RegisteredMemoryCommitHandlers | None = None
_action_policy_handlers: RegisteredActionPolicyCommitHandlers | None = None


def register_memory_commit_handlers(handlers: RegisteredMemoryCommitHandlers) -> None:
    global _memory_handlers
    _memory_handlers = handlers


def register_action_policy_commit_handlers(handlers: RegisteredActionPolicyCommitHandlers) -> None:
    global _action_policy_handlers
    _action_policy_handlers = handlers


def memory_commit_handlers() -> RegisteredMemoryCommitHandlers | None:
    return _memory_handlers


def action_policy_commit_handlers() -> RegisteredActionPolicyCommitHandlers | None:
    return _action_policy_handlers


__all__ = [
    "ActionPolicyCommitHandlerRegistration",
    "MemoryCommitHandlerRegistration",
    "RegisteredActionPolicyCommitHandlers",
    "RegisteredMemoryCommitHandlers",
    "action_policy_commit_handlers",
    "memory_commit_handlers",
    "register_action_policy_commit_handlers",
    "register_memory_commit_handlers",
]
