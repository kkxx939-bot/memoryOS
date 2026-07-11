"""Canonical scope, applicability, visibility, and subject boundaries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any
from urllib.parse import quote

CORE_SCOPE_KINDS = frozenset({"principal", "workspace", "environment", "asset", "location", "episode", "global"})
HIERARCHICAL_SCOPE_KINDS = frozenset({"asset", "location"})


class ScopeResolutionSource(str, Enum):
    """Finite provenance values for resolved scope candidates."""

    EXPLICIT = "explicit"
    EVENT = "event"
    ORIGIN = "origin"
    SCHEMA_DEFAULT = "schema_default"
    INFERRED = "inferred"


def _required(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string")
    return normalized


def _confidence(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("scope confidence must be a finite number between 0 and 1") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("scope confidence must be a finite number between 0 and 1")
    return result


def _path_key(parts: tuple[str, ...]) -> str:
    return "/".join(quote(part, safe="") for part in parts)


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
        parent_id = _required(self.parent_id, "scope parent_id") if self.parent_id is not None else None
        path = tuple(_required(item, "scope parent_path item") for item in self.parent_path)
        if parent_id is not None and not path:
            path = (parent_id,)
        elif parent_id is None and path:
            parent_id = path[-1]
        object.__setattr__(self, "parent_id", parent_id)
        object.__setattr__(self, "parent_path", path)
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        source = (
            self.source if isinstance(self.source, ScopeResolutionSource) else ScopeResolutionSource(str(self.source))
        )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "inferred", bool(self.inferred))

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
        return cls(
            namespace=str(payload.get("namespace", "memoryos")),
            kind=str(payload["kind"]),
            id=str(payload["id"]),
            parent_id=str(payload["parent_id"]) if payload.get("parent_id") else None,
            attributes=dict(payload.get("attributes", {}) or {}),
            parent_path=tuple(str(item) for item in payload.get("parent_path", []) or []),
            confidence=payload.get("confidence", 1.0),
            source=str(payload.get("source") or ScopeResolutionSource.EXPLICIT.value),
            inferred=bool(payload.get("inferred", False)),
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
        return cls(tuple(ScopeRef.from_dict(item) for item in payload.get("all_of", []) or []))


@dataclass(frozen=True)
class VisibilityPolicy:
    """Read visibility; this is deliberately not assertion authority."""

    tenant_id: str
    allowed_principal_ids: tuple[str, ...] = ()
    allowed_service_ids: tuple[str, ...] = ()
    private: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", _required(self.tenant_id, "visibility tenant_id"))
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
        return cls(
            tenant_id=str(payload.get("tenant_id") or "default"),
            allowed_principal_ids=tuple(str(item) for item in payload.get("allowed_principal_ids", []) or []),
            allowed_service_ids=tuple(str(item) for item in payload.get("allowed_service_ids", []) or []),
            private=bool(payload.get("private", False)),
        )


@dataclass(frozen=True)
class AuthorityPolicy:
    """Who may assert canonical state; separate from visibility."""

    principal_ids: tuple[str, ...] = ()
    service_ids: tuple[str, ...] = ()
    inferred: bool = False

    def __post_init__(self) -> None:
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
        return cls(
            principal_ids=tuple(str(item) for item in payload.get("principal_ids", []) or []),
            service_ids=tuple(str(item) for item in payload.get("service_ids", []) or []),
            inferred=bool(payload.get("inferred", False)),
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
        subject = payload.get("canonical_subject")
        return cls(
            applicability=ScopeSelector.from_dict(dict(payload.get("applicability", {}) or {})),
            visibility=VisibilityPolicy.from_dict(dict(payload.get("visibility", {}) or {})),
            origin_refs=tuple(ScopeRef.from_dict(item) for item in payload.get("origin_refs", []) or []),
            canonical_subject=ScopeRef.from_dict(subject) if isinstance(subject, Mapping) else None,
            authority=AuthorityPolicy.from_dict(dict(payload.get("authority", {}) or {})),
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
