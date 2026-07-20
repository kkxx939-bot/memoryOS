"""MemoryOS Python SDK 的稳定、延迟加载公开入口。

本地客户端直接组合应用运行时，HTTP 客户端访问远程服务；调用方可以使用一致的
能力名称，而无需依赖内部领域模块布局。
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_PUBLIC_ATTRS = {
    "ActionCandidate": ("policy.action_policy.model.action_policy", "ActionCandidate"),
    "ActionPolicy": ("policy.action_policy.model.action_policy", "ActionPolicy"),
    "HTTPMemoryOSClient": ("openApi.sdk.http_client", "HTTPMemoryOSClient"),
    "LocalMemoryOSClient": ("openApi.sdk.client", "LocalMemoryOSClient"),
    "MemoryOSClient": ("openApi.sdk.client", "MemoryOSClient"),
    "ProcessObservationResult": (
        "policy.action_policy.workflow.result",
        "ProcessObservationResult",
    ),
    "PredictionRequest": (
        "policy.action_policy.decision.request",
        "PredictionRequest",
    ),
    "RetrievalOptions": (
        "infrastructure.context.retrieval.query_plan",
        "RetrievalOptions",
    ),
    "RetrievalQueryPlan": (
        "infrastructure.context.retrieval.query_plan",
        "RetrievalQueryPlan",
    ),
}

if TYPE_CHECKING:
    from infrastructure.context.retrieval.query_plan import RetrievalOptions, RetrievalQueryPlan
    from openApi.sdk.client import LocalMemoryOSClient, MemoryOSClient
    from openApi.sdk.http_client import HTTPMemoryOSClient
    from policy.action_policy.decision.request import PredictionRequest
    from policy.action_policy.model.action_policy import ActionCandidate, ActionPolicy
    from policy.action_policy.workflow.result import ProcessObservationResult

__all__ = [
    "ActionCandidate",
    "ActionPolicy",
    "HTTPMemoryOSClient",
    "LocalMemoryOSClient",
    "MemoryOSClient",
    "ProcessObservationResult",
    "PredictionRequest",
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
