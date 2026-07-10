from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

CORE_SCOPE_KINDS = frozenset({"principal", "workspace", "environment", "asset", "location", "episode", "global"})


def _required(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string")
    return normalized


@dataclass(frozen=True)
class ScopeRef:
    namespace: str
    kind: str
    id: str
    parent_id: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "namespace", _required(self.namespace, "scope namespace"))
        kind = _required(self.kind, "scope kind").lower()
        if kind not in CORE_SCOPE_KINDS:
            raise ValueError(f"unsupported core scope kind: {kind}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "id", _required(self.id, "scope id"))
        if self.parent_id is not None:
            object.__setattr__(self, "parent_id", _required(self.parent_id, "scope parent_id"))
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))

    @property
    def key(self) -> str:
        return f"{self.namespace}:{self.kind}:{self.id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "kind": self.kind,
            "id": self.id,
            "parent_id": self.parent_id,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScopeRef:
        return cls(
            namespace=str(payload.get("namespace", "memoryos")),
            kind=str(payload["kind"]),
            id=str(payload["id"]),
            parent_id=str(payload["parent_id"]) if payload.get("parent_id") else None,
            attributes=dict(payload.get("attributes", {}) or {}),
        )


@dataclass(frozen=True)
class ScopeSelector:
    """A small conjunction of core scopes; it is intentionally not a rule DSL."""

    all_of: tuple[ScopeRef, ...]

    def __post_init__(self) -> None:
        if not self.all_of:
            raise ValueError("applicability scope must contain at least one scope")
        unique = {scope.key: scope for scope in self.all_of}
        object.__setattr__(self, "all_of", tuple(unique.values()))

    def to_dict(self) -> dict[str, Any]:
        return {"all_of": [scope.to_dict() for scope in self.all_of]}


@dataclass(frozen=True)
class VisibilityPolicy:
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


@dataclass(frozen=True)
class MemoryScope:
    applicability: ScopeSelector
    visibility: VisibilityPolicy
    origin_refs: tuple[ScopeRef, ...] = ()

    def __post_init__(self) -> None:
        origin = {scope.key: scope for scope in self.origin_refs}
        object.__setattr__(self, "origin_refs", tuple(origin.values()))

    def validate_tenant(self, tenant_id: str) -> None:
        if self.visibility.tenant_id != tenant_id:
            raise ValueError("visibility policy cannot cross tenant boundary")

    def to_dict(self) -> dict[str, Any]:
        return {
            "applicability": self.applicability.to_dict(),
            "visibility": self.visibility.to_dict(),
            "origin_refs": [scope.to_dict() for scope in self.origin_refs],
        }


def canonical_scope_kind(external_kind: str) -> str:
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
    attributes: Mapping[str, Any] | None = None,
) -> ScopeRef:
    return ScopeRef(
        namespace=namespace,
        kind=canonical_scope_kind(kind),
        id=identifier,
        parent_id=parent_id,
        attributes=attributes or {},
    )
