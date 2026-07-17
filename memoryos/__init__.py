"""MemoryOS stable public API.

The root package intentionally resolves public objects on first access.  This
keeps ``import memoryos`` independent from the SDK, runtime, persistence and
worker composition graph while preserving the historical public imports.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

_PUBLIC_ATTRS: dict[str, tuple[str, str]] = {
    "MemoryOSClient": ("memoryos.api.sdk.client", "MemoryOSClient"),
    "RetrievalOptions": ("memoryos.contextdb.retrieval.query_plan", "RetrievalOptions"),
    "RetrievalQueryPlan": ("memoryos.contextdb.retrieval.query_plan", "RetrievalQueryPlan"),
    "PredictionRequest": ("memoryos.prediction.model.prediction_request", "PredictionRequest"),
    "ActionPolicy": ("memoryos.action_policy.model.action_policy", "ActionPolicy"),
    "ActionCandidate": ("memoryos.action_policy.model.action_policy", "ActionCandidate"),
    "ContextDB": ("memoryos.contextdb.context_db", "ContextDB"),
}

if TYPE_CHECKING:
    from memoryos.action_policy.model.action_policy import ActionCandidate, ActionPolicy
    from memoryos.api.sdk.client import MemoryOSClient
    from memoryos.contextdb.context_db import ContextDB
    from memoryos.contextdb.retrieval.query_plan import RetrievalOptions, RetrievalQueryPlan
    from memoryos.prediction.model.prediction_request import PredictionRequest


def __getattr__(name: str) -> Any:
    target = _PUBLIC_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})

__all__ = [
    "__version__",
    "MemoryOSClient",
    "RetrievalOptions",
    "RetrievalQueryPlan",
    "PredictionRequest",
    "ActionPolicy",
    "ActionCandidate",
    "ContextDB",
]
