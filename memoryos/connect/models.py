from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ConnectType:
    AGENT = "agent"
    EMBODIED = "embodied"

    @classmethod
    def values(cls) -> set[str]:
        return {cls.AGENT, cls.EMBODIED}


class PipelineMode:
    MEMORY_ONLY = "memory_only"
    CONTEXT_REDUCTION = "context_reduction"
    ACTION_CAPABLE = "action_capable"

    @classmethod
    def values(cls) -> set[str]:
        return {cls.MEMORY_ONLY, cls.CONTEXT_REDUCTION, cls.ACTION_CAPABLE}


@dataclass(frozen=True)
class CapabilityProfile:
    can_write_memory: bool = True
    can_search_context: bool = True
    can_reduce_context: bool = True
    can_predict_behavior: bool = False
    can_generate_action: bool = False
    can_execute_action: bool = False
    can_use_external_tools: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "can_write_memory": self.can_write_memory,
            "can_search_context": self.can_search_context,
            "can_reduce_context": self.can_reduce_context,
            "can_predict_behavior": self.can_predict_behavior,
            "can_generate_action": self.can_generate_action,
            "can_execute_action": self.can_execute_action,
            "can_use_external_tools": self.can_use_external_tools,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> CapabilityProfile:
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("capabilities must be an object")
        return cls(
            can_write_memory=_strict_bool(payload, "can_write_memory", True),
            can_search_context=_strict_bool(payload, "can_search_context", True),
            can_reduce_context=_strict_bool(payload, "can_reduce_context", True),
            can_predict_behavior=_strict_bool(payload, "can_predict_behavior", False),
            can_generate_action=_strict_bool(payload, "can_generate_action", False),
            can_execute_action=_strict_bool(payload, "can_execute_action", False),
            can_use_external_tools=_strict_bool(payload, "can_use_external_tools", False),
        )


@dataclass(frozen=True)
class ConnectMetadata:
    connect_type: str = ConnectType.AGENT
    adapter_id: str = "generic_agent"
    agent_instance_id: str = ""
    run_mode: str = PipelineMode.CONTEXT_REDUCTION
    world_domain: str = "digital"
    source_kind: str = "chat"
    modality: tuple[str, ...] = ("text",)
    capabilities: CapabilityProfile = field(default_factory=CapabilityProfile)
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.connect_type not in ConnectType.values():
            raise ValueError(f"Invalid connect_type: {self.connect_type}")
        if self.run_mode not in PipelineMode.values():
            raise ValueError(f"Invalid run_mode: {self.run_mode}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "connect_type": self.connect_type,
            "adapter_id": self.adapter_id,
            "agent_instance_id": self.agent_instance_id,
            "run_mode": self.run_mode,
            "world_domain": self.world_domain,
            "source_kind": self.source_kind,
            "modality": list(self.modality),
            "capabilities": self.capabilities.to_dict(),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> ConnectMetadata:
        if payload is None:
            metadata = cls()
            metadata.validate()
            return metadata
        modality_payload: Any = payload.get("modality", ("text",))
        modality: tuple[str, ...]
        if isinstance(modality_payload, str):
            modality = (str(modality_payload),)
        else:
            modality = tuple(str(item) for item in modality_payload)
        metadata = cls(
            connect_type=str(payload.get("connect_type", ConnectType.AGENT)),
            adapter_id=str(payload.get("adapter_id", "generic_agent")),
            agent_instance_id=str(payload.get("agent_instance_id", "")),
            run_mode=str(payload.get("run_mode", PipelineMode.CONTEXT_REDUCTION)),
            world_domain=str(payload.get("world_domain", "digital")),
            source_kind=str(payload.get("source_kind", "chat")),
            modality=modality or ("text",),
            capabilities=CapabilityProfile.from_dict(payload.get("capabilities")),
            extra=dict(payload.get("extra", {})),
        )
        metadata.validate()
        return metadata

    @classmethod
    def default_agent(cls, adapter_id: str = "generic_agent") -> ConnectMetadata:
        return cls(adapter_id=adapter_id)

    @classmethod
    def action_capable_embodied(
        cls,
        adapter_id: str = "reachy_mini",
        agent_instance_id: str = "",
    ) -> ConnectMetadata:
        return cls(
            connect_type=ConnectType.EMBODIED,
            adapter_id=adapter_id,
            agent_instance_id=agent_instance_id,
            run_mode=PipelineMode.ACTION_CAPABLE,
            world_domain="physical",
            source_kind="robot",
            modality=("text", "sensor", "action"),
            capabilities=CapabilityProfile(
                can_write_memory=True,
                can_search_context=True,
                can_reduce_context=True,
                can_predict_behavior=True,
                can_generate_action=True,
                can_execute_action=True,
                can_use_external_tools=True,
            ),
        )


def _strict_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, bool):
        return value
    raise ValueError(f"capability field must be boolean: {key}")
