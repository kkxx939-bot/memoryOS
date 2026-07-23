"""本地插件和外部协议共用的 Session 输入整理。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from foundation.identity import LocalUserContext


def sanitize_ingress_messages(
    messages: list[dict[str, Any]] | None,
    context: LocalUserContext,
) -> list[dict[str, Any]]:
    """规范消息角色，并删除普通载荷伪造的 authority 字段。"""

    sanitized: list[dict[str, Any]] = []
    for raw in messages or []:
        if not isinstance(raw, dict):
            raise ValueError("messages must contain objects")
        row = dict(raw)
        role = str(row.get("role") or "assistant").strip().casefold()
        if role not in {"user", "assistant", "system", "tool"}:
            role = "assistant"
        actor_id = context.user_id if role == "user" else context.adapter_id
        metadata = dict(row.get("metadata", {}) or {}) if isinstance(row.get("metadata"), dict) else {}
        _drop_authority_fields(row)
        _drop_authority_fields(metadata)
        metadata.update({"ingress_adapter_id": context.adapter_id, "local_input": True})
        row.update({"role": role, "actor_id": actor_id, "metadata": metadata})
        sanitized.append(row)
    return sanitized


def sanitize_ingress_tool_results(
    tool_results: list[dict[str, Any]] | None,
    context: LocalUserContext,
) -> list[dict[str, Any]]:
    """把工具结果统一归属到当前本地插件适配器。"""

    sanitized: list[dict[str, Any]] = []
    for raw in tool_results or []:
        if not isinstance(raw, dict):
            raise ValueError("tool_results must contain objects")
        row = dict(raw)
        metadata = dict(row.get("metadata", {}) or {}) if isinstance(row.get("metadata"), dict) else {}
        _drop_authority_fields(row)
        _drop_authority_fields(metadata)
        metadata.update({"ingress_adapter_id": context.adapter_id, "local_input": True})
        row.update({"role": "tool", "actor_id": context.adapter_id, "metadata": metadata})
        sanitized.append(row)
    return sanitized


def sanitize_session_scope(
    scope: dict[str, Any] | None,
    context: LocalUserContext,
    *,
    project_id: str,
    session_key: str,
) -> dict[str, Any]:
    raw = dict(scope or {})
    _drop_authority_fields(raw)
    for key in ("user_id", "tenant_id", "project_id", "session_key"):
        raw.pop(key, None)
    return {
        **raw,
        "user_id": context.user_id,
        "project_id": project_id,
        "session_key": session_key,
    }


def sanitize_session_provenance(
    provenance: dict[str, Any] | None,
    context: LocalUserContext,
    *,
    native_session_id: str,
) -> dict[str, Any]:
    raw = dict(provenance or {})
    _drop_authority_fields(raw)
    for key in ("native_session_id", "user_id", "tenant_id"):
        raw.pop(key, None)
    return {
        **raw,
        "native_session_id": native_session_id,
        "user_id": context.user_id,
        "adapter_id": context.adapter_id,
    }


def local_agent_metadata(payload: Any, context: LocalUserContext) -> dict[str, Any]:
    """把插件声明限制为本地 Context Reduction 接入元数据。"""

    if payload is not None and not isinstance(payload, Mapping):
        raise ValueError("connect metadata must be an object")
    raw = dict(payload or {})
    return {
        "connect_type": "agent",
        "adapter_id": context.adapter_id,
        "agent_instance_id": str(raw.get("agent_instance_id") or ""),
        "run_mode": "context_reduction",
        "world_domain": "digital",
        "source_kind": "coding_agent",
        "modality": list(raw.get("modality") or ["text"])
        if not isinstance(raw.get("modality"), str)
        else [str(raw["modality"])],
        "capabilities": {
            "can_search_context": True,
            "can_reduce_context": True,
            "can_predict_behavior": False,
            "can_generate_action": False,
            "can_execute_action": False,
            "can_use_external_tools": False,
        },
        "extra": dict(raw.get("extra", {}) or {}) if isinstance(raw.get("extra"), Mapping) else {},
    }


def _drop_authority_fields(payload: dict[str, Any]) -> None:
    for key in (
        "actor_id",
        "actor_kind",
        "asserted_by",
        "authority",
        "effect_authority",
        "source_role",
        "subjects",
    ):
        payload.pop(key, None)


__all__ = [
    "local_agent_metadata",
    "sanitize_ingress_messages",
    "sanitize_ingress_tool_results",
    "sanitize_session_provenance",
    "sanitize_session_scope",
]
