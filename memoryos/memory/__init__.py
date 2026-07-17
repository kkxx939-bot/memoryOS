"""Stable, lazily resolved Memory domain exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_PUBLIC_ATTRS = {
    "MemoryAdmissionGate": ("memoryos.memory.admission", "MemoryAdmissionGate"),
    "MemoryExtractorBackend": ("memoryos.memory.extraction", "MemoryExtractorBackend"),
    "RuleFallbackExtractor": ("memoryos.memory.extraction", "RuleFallbackExtractor"),
    "MemoryCoolingPolicy": ("memoryos.memory.lifecycle", "MemoryCoolingPolicy"),
    "Memory": ("memoryos.memory.model", "Memory"),
    "MemoryAnchor": ("memoryos.memory.model", "MemoryAnchor"),
    "MemoryCandidate": ("memoryos.memory.model", "MemoryCandidate"),
    "MemoryKind": ("memoryos.memory.model", "MemoryKind"),
    "AdmissionDecision": ("memoryos.memory.schema", "AdmissionDecision"),
    "MemoryCandidateDraft": ("memoryos.memory.schema", "MemoryCandidateDraft"),
    "MemoryOperationGroup": ("memoryos.memory.schema", "MemoryOperationGroup"),
    "MemoryType": ("memoryos.memory.schema", "MemoryType"),
    "MemoryTypeRegistry": ("memoryos.memory.schema", "MemoryTypeRegistry"),
    "MemoryTypeSchema": ("memoryos.memory.schema", "MemoryTypeSchema"),
    "MemoryUpdater": ("memoryos.memory.service", "MemoryUpdater"),
}

if TYPE_CHECKING:
    from memoryos.memory.admission import MemoryAdmissionGate
    from memoryos.memory.extraction import MemoryExtractorBackend, RuleFallbackExtractor
    from memoryos.memory.lifecycle import MemoryCoolingPolicy
    from memoryos.memory.model import Memory, MemoryAnchor, MemoryCandidate, MemoryKind
    from memoryos.memory.schema import (
        AdmissionDecision,
        MemoryCandidateDraft,
        MemoryOperationGroup,
        MemoryType,
        MemoryTypeRegistry,
        MemoryTypeSchema,
    )
    from memoryos.memory.service import MemoryUpdater


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
    "AdmissionDecision",
    "Memory",
    "MemoryAnchor",
    "MemoryAdmissionGate",
    "MemoryCandidateDraft",
    "MemoryCandidate",
    "MemoryCoolingPolicy",
    "MemoryExtractorBackend",
    "MemoryKind",
    "MemoryOperationGroup",
    "MemoryType",
    "MemoryTypeRegistry",
    "MemoryTypeSchema",
    "MemoryUpdater",
    "RuleFallbackExtractor",
]
