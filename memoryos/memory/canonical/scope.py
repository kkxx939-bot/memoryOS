"""Canonical scope, applicability, visibility, and subject boundaries."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from types import MappingProxyType
from typing import Any
from urllib.parse import quote, unquote

CORE_SCOPE_KINDS = frozenset({"principal", "workspace", "environment", "asset", "location", "episode", "global"})
HIERARCHICAL_SCOPE_KINDS = frozenset({"asset", "location"})


class ScopeResolutionSource(str, Enum):
    """Finite provenance values for resolved scope candidates."""

    EXPLICIT = "explicit"
    EVENT = "event"
    ORIGIN = "origin"
    SCHEMA_DEFAULT = "schema_default"
    INFERRED = "inferred"


def _required(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string")
    return normalized


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("scope confidence must be a finite number between 0 and 1")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("scope confidence must be a finite number between 0 and 1")
    return result


def _path_key(parts: tuple[str, ...]) -> str:
    return "/".join(quote(part, safe="") for part in parts)


def _canonical_scope_reference_path(value: str) -> tuple[str, ...] | None:
    """Expand one canonical scope key into its complete logical ID path."""

    parts = value.split(":", 3)
    if len(parts) < 3 or parts[1] not in CORE_SCOPE_KINDS:
        return None
    if parts[2] != "path":
        identifier = ":".join(parts[2:]).strip()
        return (identifier,) if identifier else None
    if len(parts) != 4:
        raise ValueError("canonical scope parent path is malformed")
    logical = tuple(unquote(item).strip() for item in parts[3].split("/"))
    if not logical or any(not item for item in logical):
        raise ValueError("canonical scope parent path is malformed")
    return logical


def _append_logical_path(current: list[str], incoming: tuple[str, ...]) -> None:
    """Append a full parent path without duplicating an already supplied prefix."""

    overlap = 0
    maximum = min(len(current), len(incoming))
    for width in range(maximum, 0, -1):
        if tuple(current[-width:]) == incoming[:width]:
            overlap = width
            break
    current.extend(incoming[overlap:])


def _logical_parent_path(values: Sequence[str]) -> tuple[str, ...]:
    logical: list[str] = []
    for value in values:
        normalized = _required(value, "scope parent_path item")
        expanded = _canonical_scope_reference_path(normalized)
        if expanded is None:
            logical.append(normalized)
        else:
            _append_logical_path(logical, expanded)
    return tuple(logical)


def _string_array(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{name} must be an array of non-empty strings")
    return tuple(_required(item, f"{name} item") for item in value)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


@dataclass(frozen=True)
class ScopeRef:
    """A finite, provenance-bearing Identity V2 scope reference."""

    namespace: str
    kind: str
    id: str
    parent_id: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    parent_path: tuple[str, ...] = ()
    confidence: float = 1.0
    source: ScopeResolutionSource | str = ScopeResolutionSource.EXPLICIT
    inferred: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace", _required(self.namespace, "scope namespace"))
        kind = _required(self.kind, "scope kind").lower()
        if kind not in CORE_SCOPE_KINDS:
            raise ValueError(f"unsupported core scope kind: {kind}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "id", _required(self.id, "scope id"))
        parent_reference = _required(self.parent_id, "scope parent_id") if self.parent_id is not None else None
        if not isinstance(self.parent_path, Sequence) or isinstance(self.parent_path, str | bytes):
            raise ValueError("scope parent_path must be an array of non-empty strings")
        path = _logical_parent_path(self.parent_path)
        parent_reference_path = (
            _canonical_scope_reference_path(parent_reference) if parent_reference is not None else None
        )
        if parent_reference is not None and parent_reference_path is None:
            parent_reference_path = (parent_reference,)
        if parent_reference_path and path and parent_reference_path[-1] != path[-1]:
            raise ValueError("scope parent_id must equal the final parent_path item")
        if kind not in HIERARCHICAL_SCOPE_KINDS and (parent_reference is not None or path):
            raise ValueError(f"scope kind {kind} does not support parent hierarchy")
        if parent_reference_path and not path:
            path = parent_reference_path
        elif parent_reference_path and len(parent_reference_path) > 1 and path != parent_reference_path:
            raise ValueError("scope parent_id path must equal parent_path")
        parent_id = path[-1] if path else None
        object.__setattr__(self, "parent_id", parent_id)
        object.__setattr__(self, "parent_path", path)
        if not isinstance(self.attributes, Mapping):
            raise ValueError("scope attributes must be an object")
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        source = (
            self.source if isinstance(self.source, ScopeResolutionSource) else ScopeResolutionSource(str(self.source))
        )
        object.__setattr__(self, "source", source)
        if not isinstance(self.inferred, bool):
            raise ValueError("scope inferred must be a boolean")

    @property
    def hierarchy_path(self) -> tuple[str, ...]:
        return (*self.parent_path, self.id)

    @property
    def key(self) -> str:
        """A deterministic identity key including required parent hierarchy."""

        if self.kind not in HIERARCHICAL_SCOPE_KINDS or not self.parent_path:
            return f"{self.namespace}:{self.kind}:{self.id}"
        return f"{self.namespace}:{self.kind}:path:{_path_key(self.hierarchy_path)}"

    @property
    def key_candidates(self) -> tuple[str, ...]:
        """Safe read keys; parent-aware scopes never fall back ambiguously."""

        return (self.key,)

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "kind": self.kind,
            "id": self.id,
            "parent_id": self.parent_id,
            "parent_path": list(self.parent_path),
            "attributes": dict(self.attributes),
            "confidence": self.confidence,
            "source": ScopeResolutionSource(self.source).value,
            "inferred": self.inferred,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScopeRef:
        _mapping(payload, "scope")
        parent_id = payload.get("parent_id")
        if parent_id is not None and not isinstance(parent_id, str):
            raise ValueError("scope parent_id must be a non-empty string")
        attributes = payload.get("attributes", {})
        _mapping(attributes, "scope attributes")
        parent_path = _string_array(payload.get("parent_path", []), "scope parent_path")
        inferred = payload.get("inferred", False)
        if not isinstance(inferred, bool):
            raise ValueError("scope inferred must be a boolean")
        return cls(
            namespace=_required(payload.get("namespace", "memoryos"), "scope namespace"),
            kind=_required(payload.get("kind"), "scope kind"),
            id=_required(payload.get("id"), "scope id"),
            parent_id=parent_id,
            attributes=attributes,
            parent_path=parent_path,
            confidence=payload.get("confidence", 1.0),
            source=_required(
                payload.get("source", ScopeResolutionSource.EXPLICIT.value),
                "scope source",
            ),
            inferred=inferred,
        )


def scope_key_from_payload(payload: ScopeRef | Mapping[str, Any]) -> str:
    """Return the one canonical key used by identity and final filtering."""

    return payload.key if isinstance(payload, ScopeRef) else ScopeRef.from_dict(payload).key


def scope_key_candidates_from_payload(
    payload: ScopeRef | Mapping[str, Any],
) -> tuple[str, ...]:
    """Return the single safe Identity V2 key."""

    scope = payload if isinstance(payload, ScopeRef) else ScopeRef.from_dict(payload)
    return scope.key_candidates


def scope_keys_from_payloads(payloads: object) -> tuple[str, ...]:
    """Parse a complete all_of array so malformed members cannot disappear."""

    if not isinstance(payloads, Sequence) or isinstance(payloads, str | bytes):
        raise ValueError("scope applicability all_of must be an array")
    keys: list[str] = []
    for payload in payloads:
        if not isinstance(payload, Mapping):
            raise ValueError("scope applicability all_of must contain scope objects")
        keys.append(scope_key_from_payload(payload))
    return tuple(dict.fromkeys(keys))


@dataclass(frozen=True)
class ScopeSelector:
    """The applicability selector, independent of subject and visibility."""

    all_of: tuple[ScopeRef, ...]

    def __post_init__(self) -> None:
        if not self.all_of:
            raise ValueError("applicability scope must contain at least one scope")
        unique = {scope.key: scope for scope in self.all_of}
        object.__setattr__(self, "all_of", tuple(unique[key] for key in sorted(unique)))

    def to_dict(self) -> dict[str, Any]:
        return {"all_of": [scope.to_dict() for scope in self.all_of]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScopeSelector:
        _mapping(payload, "scope applicability")
        raw_all_of = payload.get("all_of")
        if not isinstance(raw_all_of, Sequence) or isinstance(raw_all_of, str | bytes):
            raise ValueError("scope applicability all_of must be an array")
        if any(not isinstance(item, Mapping) for item in raw_all_of):
            raise ValueError("scope applicability all_of must contain scope objects")
        return cls(tuple(ScopeRef.from_dict(item) for item in raw_all_of))


@dataclass(frozen=True)
class VisibilityPolicy:
    """Read visibility; this is deliberately not assertion authority."""

    tenant_id: str
    allowed_principal_ids: tuple[str, ...] = ()
    allowed_service_ids: tuple[str, ...] = ()
    private: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", _required(self.tenant_id, "visibility tenant_id"))
        if any(not isinstance(item, str) or not item.strip() for item in self.allowed_principal_ids):
            raise ValueError("visibility allowed_principal_ids must contain non-empty strings")
        if any(not isinstance(item, str) or not item.strip() for item in self.allowed_service_ids):
            raise ValueError("visibility allowed_service_ids must contain non-empty strings")
        if not isinstance(self.private, bool):
            raise ValueError("visibility private must be a boolean")
        object.__setattr__(self, "allowed_principal_ids", tuple(dict.fromkeys(self.allowed_principal_ids)))
        object.__setattr__(self, "allowed_service_ids", tuple(dict.fromkeys(self.allowed_service_ids)))

    def permits(self, *, tenant_id: str, principal_id: str | None = None, service_id: str | None = None) -> bool:
        if tenant_id != self.tenant_id:
            return False
        if not self.private and not self.allowed_principal_ids and not self.allowed_service_ids:
            return True
        return bool(
            (principal_id and principal_id in self.allowed_principal_ids)
            or (service_id and service_id in self.allowed_service_ids)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "allowed_principal_ids": list(self.allowed_principal_ids),
            "allowed_service_ids": list(self.allowed_service_ids),
            "private": self.private,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VisibilityPolicy:
        _mapping(payload, "scope visibility")
        private = payload.get("private", False)
        if not isinstance(private, bool):
            raise ValueError("visibility private must be a boolean")
        return cls(
            tenant_id=_required(payload.get("tenant_id"), "visibility tenant_id"),
            allowed_principal_ids=_string_array(
                payload.get("allowed_principal_ids", []),
                "visibility allowed_principal_ids",
            ),
            allowed_service_ids=_string_array(
                payload.get("allowed_service_ids", []),
                "visibility allowed_service_ids",
            ),
            private=private,
        )


@dataclass(frozen=True)
class AuthorityPolicy:
    """Who may assert canonical state; separate from visibility."""

    principal_ids: tuple[str, ...] = ()
    service_ids: tuple[str, ...] = ()
    inferred: bool = False

    def __post_init__(self) -> None:
        if any(not isinstance(item, str) or not item.strip() for item in self.principal_ids):
            raise ValueError("authority principal_ids must contain non-empty strings")
        if any(not isinstance(item, str) or not item.strip() for item in self.service_ids):
            raise ValueError("authority service_ids must contain non-empty strings")
        if not isinstance(self.inferred, bool):
            raise ValueError("authority inferred must be a boolean")
        object.__setattr__(self, "principal_ids", tuple(dict.fromkeys(self.principal_ids)))
        object.__setattr__(self, "service_ids", tuple(dict.fromkeys(self.service_ids)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "principal_ids": list(self.principal_ids),
            "service_ids": list(self.service_ids),
            "inferred": self.inferred,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AuthorityPolicy:
        _mapping(payload, "scope authority")
        inferred = payload.get("inferred", False)
        if not isinstance(inferred, bool):
            raise ValueError("authority inferred must be a boolean")
        return cls(
            principal_ids=_string_array(payload.get("principal_ids", []), "authority principal_ids"),
            service_ids=_string_array(payload.get("service_ids", []), "authority service_ids"),
            inferred=inferred,
        )


@dataclass(frozen=True)
class MemoryScope:
    """Canonical subject, applicability, visibility, authority, and origin."""

    applicability: ScopeSelector
    visibility: VisibilityPolicy
    origin_refs: tuple[ScopeRef, ...] = ()
    canonical_subject: ScopeRef | None = None
    authority: AuthorityPolicy = field(default_factory=AuthorityPolicy)

    def __post_init__(self) -> None:
        origin = {scope.key: scope for scope in self.origin_refs}
        object.__setattr__(self, "origin_refs", tuple(origin[key] for key in sorted(origin)))

    def validate_tenant(self, tenant_id: str) -> None:
        if self.visibility.tenant_id != tenant_id:
            raise ValueError("visibility policy cannot cross tenant boundary")

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_subject": self.canonical_subject.to_dict() if self.canonical_subject else None,
            "applicability": self.applicability.to_dict(),
            "visibility": self.visibility.to_dict(),
            "authority": self.authority.to_dict(),
            "origin_refs": [scope.to_dict() for scope in self.origin_refs],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MemoryScope:
        _mapping(payload, "memory scope")
        for field_name in ("applicability", "visibility", "authority"):
            if field_name not in payload or not isinstance(payload[field_name], Mapping):
                raise ValueError(f"memory scope {field_name} must be an object")
        raw_origin = payload.get("origin_refs", [])
        if not isinstance(raw_origin, Sequence) or isinstance(raw_origin, str | bytes):
            raise ValueError("memory scope origin_refs must be an array")
        if any(not isinstance(item, Mapping) for item in raw_origin):
            raise ValueError("memory scope origin_refs must contain scope objects")
        subject = payload.get("canonical_subject")
        if subject is not None and not isinstance(subject, Mapping):
            raise ValueError("memory scope canonical_subject must be an object or null")
        return cls(
            applicability=ScopeSelector.from_dict(payload["applicability"]),
            visibility=VisibilityPolicy.from_dict(payload["visibility"]),
            origin_refs=tuple(ScopeRef.from_dict(item) for item in raw_origin),
            canonical_subject=ScopeRef.from_dict(subject) if isinstance(subject, Mapping) else None,
            authority=AuthorityPolicy.from_dict(payload["authority"]),
        )


def canonical_scope_kind(external_kind: str) -> str:
    """Map finite external aliases to the canonical scope vocabulary."""

    aliases = {
        "person": "principal",
        "user": "principal",
        "project": "workspace",
        "repository": "workspace",
        "repo": "workspace",
        "worktree": "workspace",
        "home": "environment",
        "factory": "environment",
        "robot": "asset",
        "device": "asset",
        "room": "location",
        "zone": "location",
        "session": "episode",
    }
    normalized = str(external_kind).strip().lower()
    return aliases.get(normalized, normalized)


def scope_from_external(
    kind: str,
    identifier: str,
    *,
    namespace: str = "memoryos",
    parent_id: str | None = None,
    parent_path: tuple[str, ...] = (),
    attributes: Mapping[str, Any] | None = None,
    confidence: float = 1.0,
    source: ScopeResolutionSource | str = ScopeResolutionSource.EXPLICIT,
    inferred: bool = False,
) -> ScopeRef:
    """Convert SDK scope input without allowing arbitrary scope kinds."""

    return ScopeRef(
        namespace=namespace,
        kind=canonical_scope_kind(kind),
        id=identifier,
        parent_id=parent_id,
        attributes=attributes or {},
        parent_path=parent_path,
        confidence=confidence,
        source=source,
        inferred=inferred,
    )
