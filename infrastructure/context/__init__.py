"""上下文检索、组装、维护和召回轨迹能力的公开接口。"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from infrastructure.context.orchestrator import (
        RetrievalMetrics,
        RetrievalUnavailableError,
        UnifiedRetrievalOrchestrator,
        UnifiedRetrievalResult,
    )
    from infrastructure.context.query_service import ContextQueryService
    from infrastructure.context.trace import RecallTraceService

__all__ = [
    "ContextQueryService",
    "RetrievalMetrics",
    "RecallTraceService",
    "RetrievalUnavailableError",
    "UnifiedRetrievalOrchestrator",
    "UnifiedRetrievalResult",
]

_EXPORTS = {
    "ContextQueryService": ("infrastructure.context.query_service", "ContextQueryService"),
    "RetrievalMetrics": ("infrastructure.context.orchestrator", "RetrievalMetrics"),
    "RecallTraceService": ("infrastructure.context.trace", "RecallTraceService"),
    "RetrievalUnavailableError": ("infrastructure.context.orchestrator", "RetrievalUnavailableError"),
    "UnifiedRetrievalOrchestrator": (
        "infrastructure.context.orchestrator",
        "UnifiedRetrievalOrchestrator",
    ),
    "UnifiedRetrievalResult": ("infrastructure.context.orchestrator", "UnifiedRetrievalResult"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value
