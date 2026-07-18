"""Lazy public exports for context application services."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memoryos.application.context.assembler import ContextAssembler
    from memoryos.application.context.orchestrator import (
        RetrievalMetrics,
        RetrievalUnavailableError,
        UnifiedRetrievalOrchestrator,
        UnifiedRetrievalResult,
    )
    from memoryos.application.context.query_service import ContextQueryService
    from memoryos.application.context.retrieval_service import RetrievalService
    from memoryos.application.context.trace_erase import RecallTraceEraseBackend

__all__ = [
    "ContextAssembler",
    "ContextQueryService",
    "RetrievalMetrics",
    "RecallTraceEraseBackend",
    "RetrievalService",
    "RetrievalUnavailableError",
    "UnifiedRetrievalOrchestrator",
    "UnifiedRetrievalResult",
]

_EXPORTS = {
    "ContextAssembler": ("memoryos.application.context.assembler", "ContextAssembler"),
    "ContextQueryService": ("memoryos.application.context.query_service", "ContextQueryService"),
    "RetrievalMetrics": ("memoryos.application.context.orchestrator", "RetrievalMetrics"),
    "RetrievalService": ("memoryos.application.context.retrieval_service", "RetrievalService"),
    "RecallTraceEraseBackend": ("memoryos.application.context.trace_erase", "RecallTraceEraseBackend"),
    "RetrievalUnavailableError": ("memoryos.application.context.orchestrator", "RetrievalUnavailableError"),
    "UnifiedRetrievalOrchestrator": (
        "memoryos.application.context.orchestrator",
        "UnifiedRetrievalOrchestrator",
    ),
    "UnifiedRetrievalResult": ("memoryos.application.context.orchestrator", "UnifiedRetrievalResult"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value
