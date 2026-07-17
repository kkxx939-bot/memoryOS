"""Compatibility exports for scope primitives now owned by :mod:`memoryos.core.types`."""

from memoryos.core.types import (
    CORE_SCOPE_KINDS,
    HIERARCHICAL_SCOPE_KINDS,
    AuthorityPolicy,
    ContextScope,
    ScopeRef,
    ScopeResolutionSource,
    ScopeSelector,
    VisibilityPolicy,
    scope_key_candidates_from_payload,
    scope_key_from_payload,
    scope_keys_from_payloads,
)

__all__ = [
    "AuthorityPolicy",
    "CORE_SCOPE_KINDS",
    "ContextScope",
    "HIERARCHICAL_SCOPE_KINDS",
    "ScopeRef",
    "ScopeResolutionSource",
    "ScopeSelector",
    "VisibilityPolicy",
    "scope_key_candidates_from_payload",
    "scope_key_from_payload",
    "scope_keys_from_payloads",
]
