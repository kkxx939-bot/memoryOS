"""Lazy public exports for ContextDB retrieval contracts."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CanonicalResolutionMode",
    "ContextReranker",
    "ContextSelector",
    "EmbeddingProvider",
    "QueryPlan",
    "QueryPlanner",
    "RetrievalOptions",
    "RetrievalQueryIntent",
    "RetrievalQueryPlan",
    "RetrievalScopeViolation",
    "TokenBudgetController",
    "TrustedRetrievalScope",
    "bind_trusted_scope",
    "merge_retrieval_options",
    "retrieval_options_from_legacy",
]

_PLAN = "memoryos.contextdb.retrieval.query_plan"
_PLANNER = "memoryos.application.context.query_planner"
_EXPORTS = {
    "CanonicalResolutionMode": (_PLAN, "CanonicalResolutionMode"),
    "ContextReranker": ("memoryos.contextdb.retrieval.reranker", "ContextReranker"),
    "ContextSelector": ("memoryos.contextdb.retrieval.context_selector", "ContextSelector"),
    "EmbeddingProvider": ("memoryos.contextdb.retrieval.embedding", "EmbeddingProvider"),
    "QueryPlan": (_PLAN, "QueryPlan"),
    "QueryPlanner": (_PLANNER, "QueryPlanner"),
    "RetrievalOptions": (_PLAN, "RetrievalOptions"),
    "RetrievalQueryIntent": (_PLAN, "RetrievalQueryIntent"),
    "RetrievalQueryPlan": (_PLAN, "RetrievalQueryPlan"),
    "RetrievalScopeViolation": (_PLANNER, "RetrievalScopeViolation"),
    "TokenBudgetController": ("memoryos.contextdb.retrieval.token_budget", "TokenBudgetController"),
    "TrustedRetrievalScope": (_PLANNER, "TrustedRetrievalScope"),
    "bind_trusted_scope": (_PLANNER, "bind_trusted_scope"),
    "merge_retrieval_options": (_PLANNER, "merge_retrieval_options"),
    "retrieval_options_from_legacy": (_PLANNER, "retrieval_options_from_legacy"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value
