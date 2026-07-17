"""Stable prediction-pipeline exports resolved without eager package loading."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_PUBLIC_ATTRS = {
    "ActionContextBuilder": (
        "memoryos.prediction.pipeline.action_context_builder",
        "ActionContextBuilder",
    ),
    "ActionExecutor": ("memoryos.execution.action_executor", "ActionExecutor"),
    "ExecutionResult": ("memoryos.execution.action_executor", "ExecutionResult"),
    "Executor": ("memoryos.execution.action_executor", "Executor"),
    "ObservationNormalizer": (
        "memoryos.prediction.pipeline.observation_normalizer",
        "ObservationNormalizer",
    ),
    "PolicyGate": ("memoryos.prediction.pipeline.policy_gate", "PolicyGate"),
    "PredictionEngine": (
        "memoryos.prediction.pipeline.prediction_engine",
        "PredictionEngine",
    ),
    "PredictiveObservationProcessor": (
        "memoryos.application.prediction.observation_processor",
        "PredictiveObservationProcessor",
    ),
}

if TYPE_CHECKING:
    from memoryos.application.prediction.observation_processor import PredictiveObservationProcessor
    from memoryos.execution.action_executor import ActionExecutor, ExecutionResult, Executor
    from memoryos.prediction.pipeline.action_context_builder import ActionContextBuilder
    from memoryos.prediction.pipeline.observation_normalizer import ObservationNormalizer
    from memoryos.prediction.pipeline.policy_gate import PolicyGate
    from memoryos.prediction.pipeline.prediction_engine import PredictionEngine

__all__ = [
    "ActionContextBuilder",
    "ActionExecutor",
    "ExecutionResult",
    "Executor",
    "ObservationNormalizer",
    "PolicyGate",
    "PredictionEngine",
    "PredictiveObservationProcessor",
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
