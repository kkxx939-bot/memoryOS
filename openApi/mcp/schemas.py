"""MCP 工具描述、输入 Schema 与入口参数规范化。

本模块定义 Agent 能看到和提交的字段，并把连接元数据限制到服务端允许范围；
工具执行与权限检查由 :mod:`openApi.mcp.tools` 负责。
"""

from __future__ import annotations

from typing import Any

from openApi.mcp.config import MCPServerConfig
from openApi.mcp.errors import ToolValidationError
from openApi.memory_contract import memory_request_schema
from openApi.retrieval_contract import retrieval_options_json_schema
from pre.connect import CapabilityProfile, ConnectMetadata, ConnectType, PipelineMode

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
            "options": retrieval_options_json_schema(),
            "user_id": {"type": "string"},
            "limit": {"type": "integer"},
            "context_type": {"type": "string"},
            "context_types": {"type": "array", "items": {"type": "string"}},
            "search_scope": {"type": "string"},
            "project_id": {"type": "string"},
            "retrieval_views": {"type": "array", "items": {"type": "string"}},
            "applicability_scopes": {"type": "array", "items": {"type": "object"}},
            "record_kinds": {"type": "array", "items": {"type": "string"}},
            "document_ids": {"type": "array", "items": {"type": "string"}},
            "document_kinds": {"type": "array", "items": {"type": "string"}},
            "query_intent": {"type": "string"},
            "connect_metadata": {"type": "object"},
        },
        "required": ["query"],
    },
    "memoryos_assemble_context": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "options": retrieval_options_json_schema(),
            "user_id": {"type": "string"},
            "context_types": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "integer"},
            "search_scope": {"type": "string"},
            "project_id": {"type": "string"},
            "retrieval_views": {"type": "array", "items": {"type": "string"}},
            "applicability_scopes": {"type": "array", "items": {"type": "object"}},
            "record_kinds": {"type": "array", "items": {"type": "string"}},
            "document_ids": {"type": "array", "items": {"type": "string"}},
            "document_kinds": {"type": "array", "items": {"type": "string"}},
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
    "memoryos_adopt_memory_document": memory_request_schema("adopt"),
    "memoryos_remember": memory_request_schema("remember"),
    "memoryos_edit_memory_document": memory_request_schema("edit"),
    "memoryos_rename_memory_document": memory_request_schema("rename"),
    "memoryos_merge_memory_documents": memory_request_schema("merge"),
    "memoryos_propose_memory_consolidation": memory_request_schema("merge_propose"),
    "memoryos_resume_memory_consolidation": memory_request_schema("merge_resume"),
    "memoryos_forget": memory_request_schema("forget"),
    "memoryos_memory_history": memory_request_schema("history"),
    "memoryos_restore_memory_revision": memory_request_schema("restore"),
    "memoryos_review_memory_edit": memory_request_schema("review"),
    "memoryos_preview_memory_edit": memory_request_schema("review_preview"),
    "memoryos_archive_search": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "user_id": {"type": "string"},
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

# 短名称和显式 Context 名称共用同一份统一检索 Schema，避免两个外部入口发生漂移。
TOOL_INPUT_SCHEMAS["memoryos_search"] = TOOL_INPUT_SCHEMAS["memoryos_search_context"]
TOOL_INPUT_SCHEMAS["memoryos_assemble"] = TOOL_INPUT_SCHEMAS["memoryos_assemble_context"]

TOOL_DESCRIPTIONS: dict[str, str] = {
    "memoryos_search": "Search MemoryOS context through Unified Retrieval.",
    "memoryos_assemble": "Assemble count-bounded MemoryOS context through Unified Retrieval.",
    "memoryos_search_context": "Search MemoryOS context for a coding agent.",
    "memoryos_assemble_context": "Assemble count-bounded MemoryOS context for prompt injection.",
    "memoryos_commit_session": "Commit a sanitized agent session archive.",
    "memoryos_health": "Check MemoryOS availability.",
    "memoryos_read": "Read one exact MemoryOS URI at L0, L1, or L2.",
    "memoryos_adopt_memory_document": (
        "Explicitly adopt one safe caller-owned UNMANAGED Markdown file by relative path and exact raw digest."
    ),
    "memoryos_remember": "Commit explicit content to a managed Markdown memory document.",
    "memoryos_edit_memory_document": "CAS replace one managed Markdown memory document body.",
    "memoryos_rename_memory_document": (
        "CAS rename, with an optional same-effect body edit, while preserving the document URI and ID."
    ),
    "memoryos_merge_memory_documents": "Roll forward a bounded exact-digest multi-document merge.",
    "memoryos_propose_memory_consolidation": (
        "Seal and preview a copy-on-write exact-digest multi-document consolidation without mutating live Markdown."
    ),
    "memoryos_resume_memory_consolidation": "Resume a sealed multi-document merge after projection or restart.",
    "memoryos_forget": "Soft-forget or hard-erase one exact memory document URI.",
    "memoryos_memory_history": "List retained revisions for one memory document URI.",
    "memoryos_restore_memory_revision": "Restore one retained memory document revision with CAS.",
    "memoryos_review_memory_edit": "Approve, reject or correct one sealed document edit proposal.",
    "memoryos_preview_memory_edit": "Read the bounded proposed diff for one caller-owned edit proposal.",
    "memoryos_archive_search": "Search archived coding-agent sessions.",
    "memoryos_archive_read": "Read one exact archived session.",
    "memoryos_recall_trace": "Explain a recall trace.",
    "memoryos_connection_schema": "Describe allowed MemoryOS connection profiles.",
    "memoryos_predict": "Action-capable embodied behavior prediction. Requires action tools enabled and embodied metadata.",
    "memoryos_process_observation": "Action-capable embodied observation processing. Requires action tools enabled and embodied metadata.",
}


def tool_definitions(config: MCPServerConfig | None = None) -> list[dict[str, Any]]:
    """返回本地 MCP 当前启用的工具清单。"""

    action_tools = {"memoryos_predict", "memoryos_process_observation"}
    action_enabled = config.enable_action_tools if config is not None else False
    return [
        {
            "name": name,
            "description": TOOL_DESCRIPTIONS[name],
            "inputSchema": schema,
        }
        for name, schema in TOOL_INPUT_SCHEMAS.items()
        if action_enabled or name not in action_tools
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
    """把普通 Agent 元数据收敛到服务端允许的只读上下文能力。"""

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
    """校验具身预测工具所需的显式 action-capable 连接信息。"""

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
    """在具身元数据之上继续要求实际动作执行能力。"""

    metadata = normalize_action_metadata(payload)
    if not metadata.capabilities.can_execute_action:
        raise PermissionError("process_observation requires can_execute_action=True")
    return metadata


def connection_schema(config: MCPServerConfig) -> dict[str, Any]:
    """向 Agent 描述当前本地插件连接配置。"""

    return {
        "default_agent_profile": config.default_agent_metadata().to_dict(),
        "allowed_run_modes": [PipelineMode.CONTEXT_REDUCTION],
        "allowed_adapter_ids": list(config.allowed_adapter_ids),
        "action_tools_enabled": config.enable_action_tools,
        "local_user_id": config.user_id,
        "workspace_id": config.workspace_id,
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
        raise ToolValidationError("adapter_id does not match the configured local MCP adapter")
    if adapter_id not in config.allowed_adapter_ids:
        raise ToolValidationError(f"adapter_id is not allowed: {adapter_id}")
    return adapter_id
