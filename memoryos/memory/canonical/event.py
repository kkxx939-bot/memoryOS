from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

from memoryos.memory.canonical.scope import ScopeRef


@dataclass(frozen=True)
class ActorRef:
    kind: str
    id: str

    def __post_init__(self) -> None:
        if self.kind not in {"user", "assistant", "tool", "system", "robot", "sensor", "service"}:
            raise ValueError(f"unsupported actor kind: {self.kind}")
        if not str(self.id).strip():
            raise ValueError("actor id must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "id": self.id}


@dataclass(frozen=True)
class SubjectRef:
    kind: str
    id: str

    def __post_init__(self) -> None:
        if not str(self.kind).strip() or not str(self.id).strip():
            raise ValueError("subject kind and id must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "id": self.id}


@dataclass(frozen=True)
class OriginContext:
    world_domain: str
    connect_type: str
    adapter_id: str
    instance_id: str | None = None
    primary_scope: ScopeRef | None = None
    qualifiers: tuple[ScopeRef, ...] = ()

    def __post_init__(self) -> None:
        for name in ("world_domain", "connect_type", "adapter_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"origin {name} must be non-empty")
        unique = {scope.key: scope for scope in self.qualifiers}
        object.__setattr__(self, "qualifiers", tuple(unique.values()))

    @property
    def scope_refs(self) -> tuple[ScopeRef, ...]:
        return tuple(scope for scope in (self.primary_scope, *self.qualifiers) if scope is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_domain": self.world_domain,
            "connect_type": self.connect_type,
            "adapter_id": self.adapter_id,
            "instance_id": self.instance_id,
            "primary_scope": self.primary_scope.to_dict() if self.primary_scope else None,
            "qualifiers": [scope.to_dict() for scope in self.qualifiers],
        }


@dataclass(frozen=True)
class EventEnvelope:
    event_id: str
    event_type: str
    tenant_id: str
    actor: ActorRef
    subjects: tuple[SubjectRef, ...]
    origin: OriginContext
    episode_id: str
    session_id: str | None
    occurred_at: datetime
    content: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("event_id", "event_type", "tenant_id", "episode_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        occurred_at = self.occurred_at
        if occurred_at.tzinfo is None:
            occurred_at = occurred_at.replace(tzinfo=timezone.utc)
        object.__setattr__(self, "occurred_at", occurred_at.astimezone(timezone.utc))
        object.__setattr__(self, "subjects", tuple(dict.fromkeys(self.subjects)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, Mapping):
            for key in ("content", "text", "raw_text", "tool_output", "result"):
                if key in self.content:
                    return str(self.content[key])
        return str(self.content)
