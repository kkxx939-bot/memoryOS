"""从 Session 事实源构造的不可变证据事件。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from foundation.integrity import canonical_digest, canonical_json
from foundation.integrity.canonical_json import immutable_snapshot
from pre.evidence.model.scope import ScopeRef


def _utc(value: datetime) -> datetime:
    resolved = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return resolved.astimezone(timezone.utc)


@dataclass(frozen=True)
class ActorRef:
    kind: str
    id: str
    role: str = ""
    id_inferred: bool = False
    role_inferred: bool = False

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().casefold()
        role = str(self.role or kind).strip().casefold()
        allowed_kinds = {"user", "assistant", "tool", "system", "robot", "sensor", "service"}
        allowed_roles = {*allowed_kinds, "agent"}
        if kind not in allowed_kinds or role not in allowed_roles or not str(self.id).strip():
            raise ValueError("invalid evidence actor")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "id", str(self.id).strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "id": self.id,
            "role": self.role,
            "id_inferred": self.id_inferred,
            "role_inferred": self.role_inferred,
        }


@dataclass(frozen=True)
class SubjectRef:
    kind: str
    id: str
    inferred: bool = False

    def __post_init__(self) -> None:
        kind = str(self.kind).strip()
        identifier = str(self.id).strip()
        if not kind or not identifier:
            raise ValueError("evidence subject kind and id are required")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "id", identifier)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "id": self.id, "inferred": self.inferred}


@dataclass(frozen=True)
class OriginContext:
    world_domain: str
    connect_type: str
    adapter_id: str
    instance_id: str | None = None
    primary_scope: ScopeRef | None = None
    qualifiers: tuple[ScopeRef, ...] = ()

    def __post_init__(self) -> None:
        if not self.world_domain or not self.connect_type or not self.adapter_id:
            raise ValueError("evidence origin fields are required")
        unique = {item.key: item for item in self.qualifiers}
        object.__setattr__(self, "qualifiers", tuple(unique[key] for key in sorted(unique)))

    @property
    def scope_refs(self) -> tuple[ScopeRef, ...]:
        return tuple(item for item in (self.primary_scope, *self.qualifiers) if item is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_domain": self.world_domain,
            "connect_type": self.connect_type,
            "adapter_id": self.adapter_id,
            "instance_id": self.instance_id,
            "primary_scope": self.primary_scope.to_dict() if self.primary_scope else None,
            "qualifiers": [item.to_dict() for item in self.qualifiers],
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
    session_id: str
    occurred_at: datetime
    ingested_at: datetime
    sequence: int
    content: Any = field(repr=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)
    occurred_at_inferred: bool = False
    ingested_at_inferred: bool = False
    sequence_inferred: bool = False

    def __post_init__(self) -> None:
        if not all((self.event_id, self.event_type, self.tenant_id, self.episode_id, self.session_id)):
            raise ValueError("evidence event identity is incomplete")
        object.__setattr__(self, "occurred_at", _utc(self.occurred_at))
        object.__setattr__(self, "ingested_at", _utc(self.ingested_at))
        object.__setattr__(self, "sequence", int(self.sequence))
        unique_subjects = {
            (item.kind, item.id, item.inferred): item
            for item in self.subjects
        }
        object.__setattr__(
            self,
            "subjects",
            tuple(unique_subjects[key] for key in sorted(unique_subjects)),
        )
        object.__setattr__(self, "content", immutable_snapshot(self.content))
        object.__setattr__(self, "metadata", immutable_snapshot(self.metadata))

    def text(self) -> str:
        if isinstance(self.content, Mapping):
            # 不同接入端对事件正文使用不同字段名；统一在证据边界解析，
            # 避免下游记忆形成和 Context 投影各自维护一套字段判断。
            for key in ("content", "text", "raw_text", "tool_output", "result", "output"):
                value = self.content.get(key)
                if value is not None:
                    return value if isinstance(value, str) else canonical_json(value)
        return self.content if isinstance(self.content, str) else canonical_json(self.content)

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "tenant_id": self.tenant_id,
            "actor": self.actor.to_dict(),
            "subjects": [item.to_dict() for item in self.subjects],
            "origin": self.origin.to_dict(),
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "occurred_at": self.occurred_at,
            "ingested_at": self.ingested_at,
            "sequence": self.sequence,
            "content": self.content,
            "metadata": self.metadata,
            "inference": {
                "occurred_at": self.occurred_at_inferred,
                "ingested_at": self.ingested_at_inferred,
                "sequence": self.sequence_inferred,
            },
        }
        return {**payload, **({"event_digest": canonical_digest(payload)} if include_digest else {})}


__all__ = ["ActorRef", "EventEnvelope", "OriginContext", "SubjectRef"]
