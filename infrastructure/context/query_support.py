"""查询参数归一化与召回轨迹辅助逻辑。"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from foundation.identity import LocalUserContext
from foundation.scope import scope_key_from_payload
from infrastructure.context.query_planner import merge_retrieval_options
from infrastructure.context.retrieval.query_plan import RetrievalOptions, RetrievalQueryIntent


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
        raise ValueError("local query must select one workspace_id")
    return option_workspace_ids[0] if option_workspace_ids else None


def _merge_public_retrieval_options(
    structured: RetrievalOptions | None,
    legacy: RetrievalOptions,
    *,
    legacy_limit: int,
    legacy_limit_default: int,
    legacy_query_intent: str | None = None,
) -> RetrievalOptions:
    if structured is None:
        return legacy
    if legacy_limit != legacy_limit_default and legacy_limit != structured.final_limit:
        raise ValueError("structured options conflict with legacy limit")
    if legacy_query_intent:
        try:
            normalized_intent = RetrievalQueryIntent(str(legacy_query_intent).strip().upper())
        except ValueError as exc:
            raise ValueError(f"unknown query_intent: {legacy_query_intent!r}") from exc
        if normalized_intent != structured.query_intent:
            raise ValueError("structured options conflict with legacy query_intent")
    return merge_retrieval_options(structured, legacy)


def _prepare_retrieval_options(
    options: RetrievalOptions,
    *,
    caller: LocalUserContext | None,
    project_id: str,
) -> RetrievalOptions:
    """把本地用户与业务 Scope 转换为检索过滤条件。

    显式 Scope 只是业务过滤条件，不代表访问授权；未显式提供时才用当前用户和
    工作区补齐默认过滤范围。
    """

    metadata = dict(options.metadata_filters)
    raw_scope_keys = metadata.get("applicability_scope_keys")
    explicit_scope = raw_scope_keys is not None
    if raw_scope_keys is None:
        requested: list[str] = []
    elif isinstance(raw_scope_keys, str):
        requested = [raw_scope_keys]
    elif isinstance(raw_scope_keys, (list, tuple, set, frozenset)):
        requested = [str(item).strip() for item in raw_scope_keys]
    else:
        raise TypeError("applicability_scope_keys must be a sequence")
    if any(not item or "\x00" in item for item in requested):
        raise TypeError("applicability_scope_keys must contain non-empty strings")

    if caller is not None:
        allowed = set(caller.retrieval_scope_keys(workspace_id=project_id))
        selected = requested if explicit_scope else sorted(allowed)
    else:
        selected = list(requested)
        if options.owner_user_id:
            selected.append(f"memoryos:principal:{options.owner_user_id}")
        selected.extend(
            f"memoryos:workspace:{workspace_id}"
            for workspace_id in options.workspace_ids
            if workspace_id and workspace_id != "__memoryos_principal_only__"
        )

    if selected:
        metadata["applicability_scope_keys"] = list(dict.fromkeys(selected))
    elif explicit_scope or caller is not None:
        metadata["applicability_scope_keys"] = []
        metadata["require_unscoped"] = True
    else:
        metadata.pop("applicability_scope_keys", None)

    return replace(options, metadata_filters=metadata)


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
    "_requested_workspace",
    "_scope_keys",
    "_prepare_retrieval_options",
]
