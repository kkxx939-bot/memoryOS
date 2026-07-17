"""Stable, lazily resolved Memory infrastructure integrations."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_PUBLIC_ATTRS = {
    "BoundedCanonicalResolver": (
        "memoryos.memory.integration.canonical_resolver",
        "BoundedCanonicalResolver",
    ),
    "CanonicalCommitCoordinator": (
        "memoryos.memory.integration.coordinator",
        "CanonicalCommitCoordinator",
    ),
    "CanonicalCommitPlanning": (
        "memoryos.memory.integration.planning",
        "CanonicalCommitPlanning",
    ),
    "CanonicalMemoryCommitHandler": (
        "memoryos.memory.integration.commit_handler",
        "CanonicalMemoryCommitHandler",
    ),
    "CanonicalMemoryContextOverlay": (
        "memoryos.memory.integration.context_overlay",
        "CanonicalMemoryContextOverlay",
    ),
    "CanonicalMemoryIndexPolicy": (
        "memoryos.memory.integration.index_policy",
        "CanonicalMemoryIndexPolicy",
    ),
    "CanonicalResolutionResult": (
        "memoryos.memory.integration.canonical_resolver",
        "CanonicalResolutionResult",
    ),
    "CurrentSlotMigrationBackfill": (
        "memoryos.memory.integration.current_slot_backfill",
        "CurrentSlotMigrationBackfill",
    ),
    "bind_canonical_commit_domain_classifier": (
        "memoryos.memory.integration.commit_handler",
        "bind_canonical_commit_domain_classifier",
    ),
    "validate_canonical_authoritative_state": (
        "memoryos.memory.integration.consistency",
        "validate_canonical_authoritative_state",
    ),
}

if TYPE_CHECKING:
    from memoryos.memory.integration.canonical_resolver import (
        BoundedCanonicalResolver,
        CanonicalResolutionResult,
    )
    from memoryos.memory.integration.commit_handler import (
        CanonicalMemoryCommitHandler,
        bind_canonical_commit_domain_classifier,
    )
    from memoryos.memory.integration.consistency import validate_canonical_authoritative_state
    from memoryos.memory.integration.context_overlay import CanonicalMemoryContextOverlay
    from memoryos.memory.integration.coordinator import CanonicalCommitCoordinator
    from memoryos.memory.integration.current_slot_backfill import CurrentSlotMigrationBackfill
    from memoryos.memory.integration.index_policy import CanonicalMemoryIndexPolicy
    from memoryos.memory.integration.planning import CanonicalCommitPlanning


def __getattr__(name: str) -> Any:
    target = _PUBLIC_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})

__all__ = [
    "BoundedCanonicalResolver",
    "CanonicalCommitCoordinator",
    "CanonicalCommitPlanning",
    "CanonicalMemoryCommitHandler",
    "CanonicalMemoryContextOverlay",
    "CanonicalMemoryIndexPolicy",
    "CanonicalResolutionResult",
    "CurrentSlotMigrationBackfill",
    "bind_canonical_commit_domain_classifier",
    "validate_canonical_authoritative_state",
]
