"""Canonical memory scope composition and external alias mapping."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

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


class MemoryScope(ContextScope):
    """Canonical subject, applicability, visibility, authority, and origin."""

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MemoryScope:
        return cast(MemoryScope, super().from_dict(payload))


def canonical_scope_kind(external_kind: str) -> str:
    """Map finite external aliases to the canonical scope vocabulary."""

    aliases = {
        "person": "principal",
        "user": "principal",
        "project": "workspace",
        "repository": "workspace",
        "repo": "workspace",
        "worktree": "workspace",
        "home": "environment",
        "factory": "environment",
        "robot": "asset",
        "device": "asset",
        "room": "location",
        "zone": "location",
        "session": "episode",
    }
    normalized = str(external_kind).strip().lower()
    return aliases.get(normalized, normalized)


def scope_from_external(
    kind: str,
    identifier: str,
    *,
    namespace: str = "memoryos",
    parent_id: str | None = None,
    parent_path: tuple[str, ...] = (),
    attributes: Mapping[str, Any] | None = None,
    confidence: float = 1.0,
    source: ScopeResolutionSource | str = ScopeResolutionSource.EXPLICIT,
    inferred: bool = False,
) -> ScopeRef:
    """Convert SDK scope input without allowing arbitrary scope kinds."""

    return ScopeRef(
        namespace=namespace,
        kind=canonical_scope_kind(kind),
        id=identifier,
        parent_id=parent_id,
        attributes=attributes or {},
        parent_path=parent_path,
        confidence=confidence,
        source=source,
        inferred=inferred,
    )


__all__ = [
    "AuthorityPolicy",
    "CORE_SCOPE_KINDS",
    "HIERARCHICAL_SCOPE_KINDS",
    "MemoryScope",
    "ScopeRef",
    "ScopeResolutionSource",
    "ScopeSelector",
    "VisibilityPolicy",
    "canonical_scope_kind",
    "scope_from_external",
    "scope_key_candidates_from_payload",
    "scope_key_from_payload",
    "scope_keys_from_payloads",
]
