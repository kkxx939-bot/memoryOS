"""Shared query normalization and recall-trace helpers."""

from __future__ import annotations

import hashlib
import inspect
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from memoryos.application.context.orchestrator import UnifiedRetrievalResult
from memoryos.application.context.query_planner import TrustedRetrievalScope, merge_retrieval_options
from memoryos.contextdb.retrieval.query_plan import RetrievalOptions, RetrievalQueryIntent
from memoryos.core.clock import utc_now
from memoryos.core.durable_io import atomic_write_json
from memoryos.core.types import scope_key_from_payload
from memoryos.security.context_projection import ContextProjectionSanitizer
from memoryos.security.trusted_context import TrustedRequestContext


def _coerce_retrieval_options(value: Any) -> RetrievalOptions | None:
    if value is None:
        return None
    if isinstance(value, RetrievalOptions):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("options must be a retrieval options object")
    return RetrievalOptions.from_dict(value)

def _supported_kwargs(function: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """处理  supported kwargs 这一步。"""
    parameters = inspect.signature(function).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def _compatible_scalar(left: str | None, right: str | None, label: str) -> str | None:
    normalized_left = str(left).strip() if left is not None else ""
    normalized_right = str(right).strip() if right is not None else ""
    if normalized_left and normalized_right and normalized_left != normalized_right:
        raise ValueError(f"structured options conflict with legacy {label}")
    return normalized_left or normalized_right or None


def _requested_workspace(project_id: str, option_workspace_ids: tuple[str, ...]) -> str | None:
    requested = str(project_id or "").strip()
    if requested:
        if option_workspace_ids and option_workspace_ids != (requested,):
            raise ValueError("structured options conflict with legacy workspace_ids")
        return requested
    if len(option_workspace_ids) > 1:
        raise ValueError("trusted caller must select one workspace_id")
    return option_workspace_ids[0] if option_workspace_ids else None


def _merge_public_retrieval_options(
    structured: RetrievalOptions | None,
    legacy: RetrievalOptions,
    *,
    legacy_limit: int,
    legacy_limit_default: int,
    legacy_token_budget: int | None = None,
    legacy_token_budget_default: int | None = None,
    legacy_query_intent: str | None = None,
) -> RetrievalOptions:
    if structured is None:
        return legacy
    if legacy_limit != legacy_limit_default and legacy_limit != structured.final_limit:
        raise ValueError("structured options conflict with legacy limit")
    if (
        legacy_token_budget is not None
        and legacy_token_budget_default is not None
        and legacy_token_budget != legacy_token_budget_default
        and legacy_token_budget != structured.token_budget
    ):
        raise ValueError("structured options conflict with legacy token_budget")
    if legacy_query_intent:
        try:
            normalized_intent = RetrievalQueryIntent(str(legacy_query_intent).strip().upper())
        except ValueError as exc:
            raise ValueError(f"unknown query_intent: {legacy_query_intent!r}") from exc
        if normalized_intent != structured.query_intent:
            raise ValueError("structured options conflict with legacy query_intent")
    return merge_retrieval_options(structured, legacy)


def _trusted_retrieval_scope(
    *,
    caller: TrustedRequestContext | None,
    tenant_id: str,
    project_id: str,
    derived_scope_keys: Sequence[str] = (),
) -> TrustedRetrievalScope:
    if caller is None:
        authorized_scope_keys = None
    else:
        authorized_scope_keys = tuple(
            sorted(
                {
                    *caller.retrieval_scope_keys(workspace_id=project_id),
                    *derived_scope_keys,
                }
            )
        )
    return TrustedRetrievalScope(
        tenant_id=tenant_id,
        owner_user_id=(caller.user_id if caller is not None else None),
        workspace_ids=((project_id,) if caller is not None and project_id else None),
        adapter_id=(caller.actor_id if caller is not None else None),
        service_id=(caller.actor_id if caller is not None and caller.actor_kind == "service" else None),
        authorized_scope_keys=authorized_scope_keys,
    )


def _record_unified_recall(client: Any, result: UnifiedRetrievalResult) -> str:
    trace_id = str(uuid.uuid4())
    plan = result.plan
    metrics = result.metrics.to_dict()
    query_plan = plan.to_dict()
    query_plan.pop("semantic_query", None)
    trace = {
        "trace_id": trace_id,
        "created_at": utc_now(),
        "query_digest": hashlib.sha256(plan.semantic_query.encode("utf-8")).hexdigest(),
        "query_utf8_bytes": len(plan.semantic_query.encode("utf-8")),
        "query_plan": query_plan,
        "scope": {
            "tenant_id": plan.tenant_id,
            "user_id": plan.owner_user_id,
            "project_id": plan.workspace_ids[0] if len(plan.workspace_ids) == 1 else "",
            "workspace_ids": list(plan.workspace_ids),
            "session_ids": list(plan.session_ids),
            "adapter_id": plan.adapter_id,
            "search_scope": plan.legacy_search_scope,
        },
        "retrieval_views": list(plan.legacy_retrieval_views),
        "metadata_filters": dict(plan.metadata_filters),
        **metrics,
        "candidate_count": metrics["fusion_candidates"],
        "selected": [
            {
                "uri": item.get("uri"),
                "source_uri": item.get("source_uri"),
                "score": item.get("score"),
                "layer": item.get("selected_layer") or item.get("layer"),
                "source_validation_status": item.get("source_validation_status"),
                "projection_lag": item.get("projection_lag"),
                "degraded_mode": item.get("degraded_mode"),
            }
            for item in result.contexts
        ],
        "dropped": [dict(item) for item in result.dropped_contexts],
        "token_budget": plan.token_budget,
        "degraded_modes": list(result.degraded_modes),
        "reranker_fallback": result.reranker_fallback,
    }
    safe_trace = ContextProjectionSanitizer().sanitize_trace(trace)
    if not isinstance(safe_trace, dict):
        raise ValueError("recall trace sanitization produced an invalid payload")
    root = _trace_root(client)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        root.chmod(0o700)
    except OSError as exc:
        raise PermissionError("recall trace directory permissions could not be secured") from exc
    atomic_write_json(root / f"{trace_id}.json", safe_trace, artifact_root=root)
    return trace_id


def _trace_root(client: Any) -> Path:
    root = Path(str(getattr(client, "root", "/tmp/memoryos-test")))
    tenant_id = str(getattr(client, "tenant_id", "default"))
    return root / "recall-traces" if tenant_id == "default" else root / "tenants" / tenant_id / "recall-traces"


def _scope_keys(
    scopes: list[dict[str, Any]] | None,
) -> list[str]:
    keys = []
    for scope in scopes or []:
        if not isinstance(scope, dict) or not scope.get("kind") or not scope.get("id"):
            raise ValueError("applicability_scopes must contain scope objects with kind and id")
        keys.append(scope_key_from_payload(scope))
    return list(dict.fromkeys(keys))



__all__ = [
    "_coerce_retrieval_options",
    "_compatible_scalar",
    "_merge_public_retrieval_options",
    "_record_unified_recall",
    "_requested_workspace",
    "_scope_keys",
    "_trace_root",
    "_trusted_retrieval_scope",
]
