"""Stable, lazily resolved SDK exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_PUBLIC_ATTRS = {
    "HTTPMemoryOSClient": ("memoryos.api.sdk.http_client", "HTTPMemoryOSClient"),
    "LocalMemoryOSClient": ("memoryos.api.sdk.client", "LocalMemoryOSClient"),
    "MemoryOSClient": ("memoryos.api.sdk.client", "MemoryOSClient"),
    "ProcessObservationResult": (
        "memoryos.application.prediction.result",
        "ProcessObservationResult",
    ),
    "RetrievalOptions": (
        "memoryos.contextdb.retrieval.query_plan",
        "RetrievalOptions",
    ),
    "RetrievalQueryPlan": (
        "memoryos.contextdb.retrieval.query_plan",
        "RetrievalQueryPlan",
    ),
}

if TYPE_CHECKING:
    from memoryos.api.sdk.client import LocalMemoryOSClient, MemoryOSClient
    from memoryos.api.sdk.http_client import HTTPMemoryOSClient
    from memoryos.application.prediction.result import ProcessObservationResult
    from memoryos.contextdb.retrieval.query_plan import RetrievalOptions, RetrievalQueryPlan

__all__ = [
    "HTTPMemoryOSClient",
    "LocalMemoryOSClient",
    "MemoryOSClient",
    "ProcessObservationResult",
    "RetrievalOptions",
    "RetrievalQueryPlan",
]


def __getattr__(name: str) -> Any:
    target = _PUBLIC_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
