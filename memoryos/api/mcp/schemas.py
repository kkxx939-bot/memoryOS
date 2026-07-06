from __future__ import annotations

from typing import Any

from memoryos.api.mcp.config import MCPServerConfig
from memoryos.api.mcp.errors import ToolValidationError
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


def required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolValidationError(f"requires non-empty string field: {key}")
    return value


def optional_int(payload: dict[str, Any], key: str, default: int, *, minimum: int = 0, maximum: int = 100_000) -> int:
    value = payload.get(key, default)
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


def normalize_agent_metadata(payload: dict[str, Any] | None, config: MCPServerConfig) -> dict[str, Any]:
    raw = _filtered_metadata(payload)
    adapter_id = str(raw.get("adapter_id") or config.adapter_id)
    if adapter_id not in config.allowed_adapter_ids:
        adapter_id = config.adapter_id
    extra: dict[str, Any] = dict(raw.get("extra", {})) if isinstance(raw.get("extra"), dict) else {}
    metadata = ConnectMetadata(
        connect_type=ConnectType.AGENT,
        adapter_id=adapter_id,
        agent_instance_id=str(raw.get("agent_instance_id", "")),
        run_mode=PipelineMode.CONTEXT_REDUCTION,
        world_domain="digital",
        source_kind="coding_agent",
        modality=tuple(str(item) for item in raw.get("modality", ("text",))) if not isinstance(raw.get("modality"), str) else (str(raw.get("modality")),),
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
        raise PermissionError("action tools require embodied/action_capable metadata with behavior prediction capability")
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
        "embodied_profile_example": ConnectMetadata.action_capable_embodied("reachy_mini").to_dict(),
    }


def _filtered_metadata(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ToolValidationError("connect_metadata must be an object")
    return {key: value for key, value in payload.items() if key in ALLOWED_METADATA_FIELDS}
