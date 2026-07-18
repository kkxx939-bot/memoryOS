"""Generic evidence scope normalization, independent of memory storage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from memoryos.core.types import ScopeRef, ScopeResolutionSource


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
    normalized = aliases.get(str(kind).strip().casefold(), str(kind).strip().casefold())
    return ScopeRef(
        namespace=namespace,
        kind=normalized,
        id=identifier,
        parent_id=parent_id,
        parent_path=parent_path,
        attributes=attributes or {},
        confidence=confidence,
        source=source,
        inferred=inferred,
    )


__all__ = ["ScopeRef", "ScopeResolutionSource", "scope_from_external"]
