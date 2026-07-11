"""Immutable, canonical event envelopes used by the evidence pipeline."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any

from memoryos.memory.canonical.scope import ScopeRef

EVENT_ENVELOPE_SCHEMA_VERSION = "event_envelope_v2"


class CanonicalSerializationError(ValueError):
    """Raised when evidence cannot be represented deterministically."""


def canonicalize(value: Any) -> Any:
    """Return a JSON-safe deterministic snapshot of an evidence value."""

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalSerializationError("non-finite floats are not valid evidence values")
        return value
    if isinstance(value, Enum):
        return canonicalize(value.value)
    if isinstance(value, datetime):
        resolved = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return resolved.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if key in result:
                raise CanonicalSerializationError(f"mapping keys collide after string normalization: {key!r}")
            result[key] = canonicalize(raw_value)
        return {key: result[key] for key in sorted(result)}
    if isinstance(value, list | tuple):
        return [canonicalize(item) for item in value]
    if isinstance(value, set | frozenset):
        items = [canonicalize(item) for item in value]
        return sorted(items, key=lambda item: canonical_json(item))
    raise CanonicalSerializationError(f"unsupported evidence value type: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def immutable_snapshot(value: Any) -> Any:
    """Deeply snapshot mutable caller data without retaining shared references."""

    normalized = canonicalize(value)
    return _freeze_normalized(normalized)


def _freeze_normalized(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_normalized(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_normalized(item) for item in value)
    return value


def _utc(value: datetime) -> datetime:
    resolved = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return resolved.astimezone(timezone.utc)


def _content_path(content: Any) -> str:
    if isinstance(content, Mapping):
        for key in ("content", "text", "raw_text", "tool_output", "result", "output"):
            if key in content:
                return f"$.{key}"
    return "$"


def resolve_content_path(content: Any, path: str) -> Any:
    """Resolve the small, non-executable JSON path subset used by EvidenceRef."""

    if path == "$":
        return content
    if not path.startswith("$."):
        raise ValueError(f"unsupported evidence content path: {path}")
    current = content
    for segment in path[2:].split("."):
        if not segment or not isinstance(current, Mapping) or segment not in current:
            raise ValueError(f"evidence content path does not exist: {path}")
        current = current[segment]
    return current


@dataclass(frozen=True)
class ActorRef:
    kind: str
    id: str
    role: str | None = None
    id_inferred: bool = False
    role_inferred: bool = False

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind not in {"user", "assistant", "tool", "system", "robot", "sensor", "service"}:
            raise ValueError(f"unsupported actor kind: {kind}")
        identifier = str(self.id).strip()
        if not identifier:
            raise ValueError("actor id must be non-empty")
        role = str(self.role or kind).strip().lower()
        if role not in {"user", "assistant", "agent", "tool", "system", "robot", "sensor", "service"}:
            raise ValueError(f"unsupported actor role: {role}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "id", identifier)
        object.__setattr__(self, "role", role)

    @property
    def inferred(self) -> bool:
        return self.id_inferred or self.role_inferred

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
            raise ValueError("subject kind and id must be non-empty")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "id", identifier)

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.id}"

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
        for name in ("world_domain", "connect_type", "adapter_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"origin {name} must be non-empty")
        unique = {scope.key: scope for scope in self.qualifiers}
        object.__setattr__(self, "qualifiers", tuple(unique[key] for key in sorted(unique)))

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
    schema_version: str = EVENT_ENVELOPE_SCHEMA_VERSION
    ingested_at: datetime | None = None
    sequence: int = 0
    occurred_at_inferred: bool = False
    ingested_at_inferred: bool = False
    sequence_inferred: bool = False
    content_path: str = ""

    def __post_init__(self) -> None:
        for name in ("schema_version", "event_id", "event_type", "tenant_id", "episode_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        occurred_at = _utc(self.occurred_at)
        ingested_at = _utc(self.ingested_at or occurred_at)
        try:
            sequence = int(self.sequence)
        except (TypeError, ValueError) as exc:
            raise ValueError("event sequence must be an integer") from exc
        content = immutable_snapshot(self.content)
        metadata = immutable_snapshot(self.metadata)
        path = str(self.content_path or _content_path(content))
        resolve_content_path(content, path)
        object.__setattr__(self, "occurred_at", occurred_at)
        object.__setattr__(self, "ingested_at", ingested_at)
        object.__setattr__(self, "sequence", sequence)
        unique_subjects: dict[tuple[str, str], SubjectRef] = {}
        for subject in self.subjects:
            key = (subject.kind, subject.id)
            if key not in unique_subjects or (unique_subjects[key].inferred and not subject.inferred):
                unique_subjects[key] = subject
        object.__setattr__(self, "subjects", tuple(unique_subjects[key] for key in sorted(unique_subjects)))
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "content_path", path)

    def content_value(self, path: str | None = None) -> Any:
        return resolve_content_path(self.content, path or self.content_path)

    def text(self, path: str | None = None) -> str:
        value = self.content_value(path)
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping | tuple):
            return canonical_json(value)
        return str(value)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tenant_id": self.tenant_id,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "actor": self.actor.to_dict(),
            "subjects": [subject.to_dict() for subject in self.subjects],
            "origin": self.origin.to_dict(),
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "occurred_at": self.occurred_at,
            "ingested_at": self.ingested_at,
            "sequence": self.sequence,
            "content_path": self.content_path,
            "content": self.content,
            "metadata": self.metadata,
            "inference": {
                "occurred_at": self.occurred_at_inferred,
                "ingested_at": self.ingested_at_inferred,
                "sequence": self.sequence_inferred,
                "actor_id": self.actor.id_inferred,
                "actor_role": self.actor.role_inferred,
                "subjects": any(subject.inferred for subject in self.subjects),
            },
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**canonicalize(self.canonical_payload()), "event_digest": self.digest}
