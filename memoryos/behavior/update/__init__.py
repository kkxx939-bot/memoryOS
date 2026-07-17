"""Stable, lazily resolved behavior-update exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PUBLIC_ATTRS = {
    "BehaviorCaseWriter": (
        "memoryos.behavior.update.behavior_case_writer",
        "BehaviorCaseWriter",
    ),
    "BehaviorClusterUpdater": (
        "memoryos.behavior.update.behavior_cluster_updater",
        "BehaviorClusterUpdater",
    ),
    "BehaviorCoolingService": (
        "memoryos.behavior.update.behavior_cooling",
        "BehaviorCoolingService",
    ),
    "BehaviorLifecycleResult": (
        "memoryos.application.memory.behavior_lifecycle",
        "BehaviorLifecycleResult",
    ),
    "BehaviorLifecycleService": (
        "memoryos.application.memory.behavior_lifecycle",
        "BehaviorLifecycleService",
    ),
    "BehaviorPatternUpdater": (
        "memoryos.behavior.update.behavior_pattern_updater",
        "BehaviorPatternUpdater",
    ),
    "OpportunityAwareDecay": (
        "memoryos.behavior.update.opportunity_decay",
        "OpportunityAwareDecay",
    ),
}

__all__ = list(_PUBLIC_ATTRS)


def __getattr__(name: str) -> Any:
    target = _PUBLIC_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
