"""Stable lazy exports for user-editable Markdown memory."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PUBLIC_ATTRS = {
    "DocumentChangeEvent": ("memoryos.memory.documents", "DocumentChangeEvent"),
    "DocumentCommitResult": ("memoryos.memory.documents", "DocumentCommitResult"),
    "DocumentEditKind": ("memoryos.memory.documents", "DocumentEditKind"),
    "DocumentEditPlan": ("memoryos.memory.documents", "DocumentEditPlan"),
    "MemoryCandidateKind": ("memoryos.memory.documents", "MemoryCandidateKind"),
    "MemoryCandidateRegistry": ("memoryos.memory.schema", "MemoryCandidateRegistry"),
    "MemoryCandidateSchema": ("memoryos.memory.schema", "MemoryCandidateSchema"),
    "MemoryDocument": ("memoryos.memory.documents", "MemoryDocument"),
    "MemoryDocumentCommitter": ("memoryos.memory.documents", "MemoryDocumentCommitter"),
    "MemoryDocumentKind": ("memoryos.memory.documents", "MemoryDocumentKind"),
    "MemoryDocumentPlanner": ("memoryos.memory.documents", "MemoryDocumentPlanner"),
    "MemoryEditProposal": ("memoryos.memory.documents", "MemoryEditProposal"),
    "MemoryExtractorBackend": ("memoryos.memory.extraction", "MemoryExtractorBackend"),
    "RuleFallbackExtractor": ("memoryos.memory.extraction", "RuleFallbackExtractor"),
}


def __getattr__(name: str) -> Any:
    target = _PUBLIC_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


__all__ = sorted(_PUBLIC_ATTRS)
