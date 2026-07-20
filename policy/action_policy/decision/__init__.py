"""ActionPolicy 在线预测与安全决策的惰性公开接口。"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_EXPORTS = {
    "ActionContext": ("policy.action_policy.decision.action_context", "ActionContext"),
    "ActionContextBuilder": ("policy.action_policy.decision.context_builder", "ActionContextBuilder"),
    "DecisionLedger": ("policy.action_policy.decision.ledger", "DecisionLedger"),
    "ObservationNormalizer": (
        "policy.action_policy.decision.observation_normalizer",
        "ObservationNormalizer",
    ),
    "PolicyDecision": ("policy.action_policy.decision.result", "PolicyDecision"),
    "PolicyGate": ("policy.action_policy.decision.gate", "PolicyGate"),
    "PredictionEngine": ("policy.action_policy.decision.engine", "PredictionEngine"),
    "PredictionRequest": ("policy.action_policy.decision.request", "PredictionRequest"),
    "PredictionResult": ("policy.action_policy.decision.result", "PredictionResult"),
}

if TYPE_CHECKING:
    from policy.action_policy.decision.action_context import ActionContext
    from policy.action_policy.decision.context_builder import ActionContextBuilder
    from policy.action_policy.decision.engine import PredictionEngine
    from policy.action_policy.decision.gate import PolicyGate
    from policy.action_policy.decision.ledger import DecisionLedger
    from policy.action_policy.decision.observation_normalizer import ObservationNormalizer
    from policy.action_policy.decision.request import PredictionRequest
    from policy.action_policy.decision.result import PolicyDecision, PredictionResult

__all__ = [
    "ActionContext",
    "ActionContextBuilder",
    "DecisionLedger",
    "ObservationNormalizer",
    "PolicyDecision",
    "PolicyGate",
    "PredictionEngine",
    "PredictionRequest",
    "PredictionResult",
]


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value
