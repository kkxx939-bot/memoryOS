"""MCP 参数结构。"""

from __future__ import annotations

from typing import Any

from memoryos.api.mcp.config import MCPServerConfig
from memoryos.api.mcp.errors import ToolValidationError
from memoryos.api.trusted_context import AUTHORITATIVE_FORGET, AUTHORITATIVE_REMEMBER
from memoryos.connect import CapabilityProfile, ConnectMetadata, ConnectType, PipelineMode

ALLOWED_METADATA_FIELDS = {
    "connect_type",
    "adapter_id",
    "agent_instance_id",
    "run_mode",
    "world_domain",
    "source_kind",
    "modality",
    "capabilities",
    "capability_profile",
    "extra",
}

ACTION_ADAPTERS = {"reachy_mini"}

TOOL_INPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    "memoryos_search_context": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "user_id": {"type": "string"},
            "limit": {"type": "integer"},
            "context_type": {"type": "string"},
            "context_types": {"type": "array", "items": {"type": "string"}},
            "search_scope": {"type": "string"},
            "project_id": {"type": "string"},
            "retrieval_views": {"type": "array", "items": {"type": "string"}},
            "tenant_id": {"type": "string"},
            "applicability_scopes": {"type": "array", "items": {"type": "object"}},
            "memory_states": {"type": "array", "items": {"type": "string"}},
            "memory_types": {"type": "array", "items": {"type": "string"}},
            "claim_uris": {"type": "array", "items": {"type": "string"}},
            "slot_uris": {"type": "array", "items": {"type": "string"}},
            "query_intent": {"type": "string"},
            "connect_metadata": {"type": "object"},
        },
        "required": ["query"],
    },
    "memoryos_assemble_context": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "user_id": {"type": "string"},
            "token_budget": {"type": "integer"},
            "context_types": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "integer"},
            "search_scope": {"type": "string"},
            "project_id": {"type": "string"},
            "retrieval_views": {"type": "array", "items": {"type": "string"}},
            "tenant_id": {"type": "string"},
            "applicability_scopes": {"type": "array", "items": {"type": "object"}},
            "memory_states": {"type": "array", "items": {"type": "string"}},
            "memory_types": {"type": "array", "items": {"type": "string"}},
            "claim_uris": {"type": "array", "items": {"type": "string"}},
            "slot_uris": {"type": "array", "items": {"type": "string"}},
            "query_intent": {"type": "string"},
            "connect_metadata": {"type": "object"},
        },
        "required": ["query"],
    },
    "memoryos_commit_session": {
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "session_id": {"type": "string"},
            "messages": {"type": "array", "items": {"type": "object"}},
            "used_contexts": {"type": "array", "items": {"type": "object"}},
            "tool_results": {"type": "array", "items": {"type": "object"}},
            "connect_metadata": {"type": "object"},
            "async_commit": {"type": "boolean"},
            "project_id": {"type": "string"},
            "session_key": {"type": "string"},
            "scope": {"type": "object"},
            "provenance": {"type": "object"},
        },
        "required": ["session_id"],
    },
    "memoryos_health": {"type": "object", "properties": {}, "required": []},
    "memoryos_read": {
        "type": "object",
        "properties": {"uri": {"type": "string"}, "layer": {"type": "string"}},
        "required": ["uri"],
    },
    "memoryos_remember": {
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "content": {"type": "string"},
            "title": {"type": "string"},
            "memory_type": {"type": "string"},
            "project_id": {"type": "string"},
            "constraint_polarity": {"type": "string"},
            "condition": {"type": "string"},
            "exception": {"type": "string"},
            "connect_metadata": {"type": "object"},
        },
        "required": ["content"],
    },
    "memoryos_forget": {
        "type": "object",
        "properties": {"user_id": {"type": "string"}, "uri": {"type": "string"}},
        "required": ["uri"],
    },
    "memoryos_archive_search": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "user_id": {"type": "string"},
            "tenant_id": {"type": "string"},
            "project_id": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["query"],
    },
    "memoryos_archive_read": {
        "type": "object",
        "properties": {"archive_uri": {"type": "string"}},
        "required": ["archive_uri"],
    },
    "memoryos_recall_trace": {
        "type": "object",
        "properties": {"trace_id": {"type": "string"}},
        "required": ["trace_id"],
    },
    "memoryos_connection_schema": {"type": "object", "properties": {}, "required": []},
    "memoryos_predict": {
        "type": "object",
        "properties": {
            "request": {"type": "object"},
            "policies": {"type": "array", "items": {"type": "object"}},
            "connect_metadata": {"type": "object"},
        },
        "required": ["request"],
    },
    "memoryos_process_observation": {
        "type": "object",
        "properties": {
            "request": {"type": "object"},
            "policies": {"type": "array", "items": {"type": "object"}},
            "connect_metadata": {"type": "object"},
            "archive_session": {"type": "boolean"},
            "async_commit": {"type": "boolean"},
        },
        "required": ["request"],
    },
}

TOOL_DESCRIPTIONS: dict[str, str] = {
    "memoryos_search_context": "Search MemoryOS context for a coding agent.",
    "memoryos_assemble_context": "Assemble token-bounded MemoryOS context for prompt injection.",
    "memoryos_commit_session": "Commit a sanitized agent session archive.",
    "memoryos_health": "Check MemoryOS availability.",
    "memoryos_read": "Read one exact MemoryOS URI at L0, L1, or L2.",
    "memoryos_remember": "Store an explicit confirmed memory.",
    "memoryos_forget": "Forget one exact MemoryOS URI.",
    "memoryos_archive_search": "Search archived coding-agent sessions.",
    "memoryos_archive_read": "Read one exact archived session.",
    "memoryos_recall_trace": "Explain a recall trace.",
    "memoryos_connection_schema": "Describe allowed MemoryOS connection profiles.",
    "memoryos_predict": "Action-capable embodied behavior prediction. Requires action tools enabled and embodied metadata.",
    "memoryos_process_observation": "Action-capable embodied observation processing. Requires action tools enabled and embodied metadata.",
}


def tool_definitions(config: MCPServerConfig | None = None) -> list[dict[str, Any]]:
    action_tools = {"memoryos_predict", "memoryos_process_observation"}
    authoritative_tools = {
        "memoryos_remember": AUTHORITATIVE_REMEMBER,
        "memoryos_forget": AUTHORITATIVE_FORGET,
    }
    action_enabled = config.enable_action_tools if config is not None else False
    capabilities = config.capabilities if config is not None else frozenset()
    return [
        {
            "name": name,
            "description": TOOL_DESCRIPTIONS[name],
            "inputSchema": schema,
        }
        for name, schema in TOOL_INPUT_SCHEMAS.items()
        if (action_enabled or name not in action_tools)
        and (name not in authoritative_tools or authoritative_tools[name] in capabilities)
    ]


def required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolValidationError(f"requires non-empty string field: {key}")
    return value


def optional_int(payload: dict[str, Any], key: str, default: int, *, minimum: int = 0, maximum: int = 100_000) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool):
        raise ToolValidationError(f"requires integer field: {key}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ToolValidationError(f"requires integer field: {key}") from exc
    if parsed < minimum or parsed > maximum:
        raise ToolValidationError(f"field out of range: {key}")
    return parsed


def optional_list(payload: dict[str, Any], key: str) -> list[Any] | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, list):
        raise ToolValidationError(f"requires array field: {key}")
    return value


def optional_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    raise ToolValidationError(f"requires boolean field: {key}")


def normalize_agent_metadata(payload: dict[str, Any] | None, config: MCPServerConfig) -> dict[str, Any]:
    raw = _filtered_metadata(payload)
    adapter_id = _allowed_agent_adapter_id(raw.get("adapter_id") or config.adapter_id, config)
    extra: dict[str, Any] = dict(raw.get("extra", {})) if isinstance(raw.get("extra"), dict) else {}
    metadata = ConnectMetadata(
        connect_type=ConnectType.AGENT,
        adapter_id=adapter_id,
        agent_instance_id=str(raw.get("agent_instance_id", "")),
        run_mode=PipelineMode.CONTEXT_REDUCTION,
        world_domain="digital",
        source_kind="coding_agent",
        modality=tuple(str(item) for item in raw.get("modality", ("text",)))
        if not isinstance(raw.get("modality"), str)
        else (str(raw.get("modality")),),
        capabilities=CapabilityProfile(
            can_write_memory=True,
            can_search_context=True,
            can_reduce_context=True,
            can_predict_behavior=False,
            can_generate_action=False,
            can_execute_action=False,
            can_use_external_tools=False,
        ),
        extra=extra,
    )
    return metadata.to_dict()


def agent_search_filter_metadata(payload: dict[str, Any] | None, config: MCPServerConfig) -> dict[str, Any]:
    raw = _filtered_metadata(payload)
    adapter_id = _allowed_agent_adapter_id(raw.get("adapter_id") or config.adapter_id, config)
    filters: dict[str, Any] = {"adapter_id": adapter_id}
    for key in ("connect_type", "run_mode", "world_domain", "source_kind"):
        value = raw.get(key)
        if isinstance(value, str) and value:
            filters[key] = value
    return filters


def normalize_action_metadata(payload: dict[str, Any] | None) -> ConnectMetadata:
    raw = _filtered_metadata(payload)
    if "capability_profile" in raw and "capabilities" not in raw:
        raw["capabilities"] = raw.pop("capability_profile")
    metadata = ConnectMetadata.from_dict(raw)
    if (
        metadata.connect_type != ConnectType.EMBODIED
        or metadata.run_mode != PipelineMode.ACTION_CAPABLE
        or metadata.adapter_id not in ACTION_ADAPTERS
        or not metadata.capabilities.can_predict_behavior
    ):
        raise PermissionError(
            "action tools require embodied/action_capable metadata with behavior prediction capability"
        )
    return metadata


def require_process_observation_metadata(payload: dict[str, Any] | None) -> ConnectMetadata:
    metadata = normalize_action_metadata(payload)
    if not metadata.capabilities.can_execute_action:
        raise PermissionError("process_observation requires can_execute_action=True")
    return metadata


def connection_schema(config: MCPServerConfig) -> dict[str, Any]:
    return {
        "default_agent_profile": config.default_agent_metadata().to_dict(),
        "allowed_run_modes": [PipelineMode.CONTEXT_REDUCTION],
        "allowed_adapter_ids": list(config.allowed_adapter_ids),
        "action_tools_enabled": config.enable_action_tools,
        "trusted_user_id": config.user_id,
        "trusted_tenant_id": config.tenant_id,
        "trusted_capabilities": sorted(config.capabilities),
        "trusted_workspace_ids": sorted(config.allowed_workspace_ids),
        "embodied_profile_example": ConnectMetadata.action_capable_embodied("reachy_mini").to_dict(),
    }


def _filtered_metadata(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ToolValidationError("connect_metadata must be an object")
    return {key: value for key, value in payload.items() if key in ALLOWED_METADATA_FIELDS}


def _allowed_agent_adapter_id(value: Any, config: MCPServerConfig) -> str:
    adapter_id = str(value)
    if adapter_id != config.adapter_id:
        raise ToolValidationError("adapter_id does not match the trusted MCP adapter")
    if adapter_id not in config.allowed_adapter_ids:
        raise ToolValidationError(f"adapter_id is not allowed: {adapter_id}")
    return adapter_id
