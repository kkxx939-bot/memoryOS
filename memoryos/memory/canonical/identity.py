"""Versioned, deterministic canonical memory identity."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any

from memoryos.core.ids import stable_hash
from memoryos.memory.canonical.proposal import MemorySemanticProposal
from memoryos.memory.canonical.scope import HIERARCHICAL_SCOPE_KINDS, MemoryScope, ScopeRef
from memoryos.memory.schema import MemoryTypeRegistry, MemoryTypeSchema

IDENTITY_ALGORITHM_V2 = "identity_v2"


def canonical_text(value: Any) -> str:
    """Normalize scalar identity text without depending on presentation case."""

    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return re.sub(r"[\s_-]+", "-", text).strip("-")


def canonical_identity_value(value: Any) -> Any:
    """Recursively canonicalize identity values with deterministic ordering."""

    if isinstance(value, Enum):
        return canonical_identity_value(value.value)
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("identity values cannot contain non-finite numbers")
        return value
    if isinstance(value, datetime):
        normalized = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return normalized.astimezone(timezone.utc).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return canonical_text(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            normalized_key = str(key)
            if normalized_key in result:
                raise ValueError(f"identity mapping contains colliding key: {normalized_key}")
            result[normalized_key] = canonical_identity_value(item)
        return result
    if isinstance(value, set | frozenset):
        canonical = [canonical_identity_value(item) for item in value]
        return sorted(canonical, key=canonical_identity_json)
    if isinstance(value, list | tuple):
        return [canonical_identity_value(item) for item in value]
    return canonical_text(value)


def canonical_identity_json(value: Any) -> str:
    return json.dumps(canonical_identity_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class AliasRegistry:
    """Deterministic value and scope aliases used before hashing Identity V2."""

    def __init__(
        self,
        aliases: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        payload = dict(aliases or {})
        self._aliases: dict[str, dict[str, str]] = {}
        for namespace, values in payload.items():
            self._aliases[str(namespace)] = {
                canonical_text(alias): str(identifier) for alias, identifier in values.items()
            }

    def resolve(self, namespace: str, value: Any) -> str:
        normalized = canonical_text(value)
        return self._aliases.get(namespace, {}).get(normalized, normalized)

    def canonical_scope(self, scope: ScopeRef) -> ScopeRef:
        identifier = self.resolve(f"scope:{scope.kind}", scope.id)
        parent_path = tuple(self.resolve(f"scope:{scope.kind}:parent", item) for item in scope.parent_path)
        return ScopeRef(
            namespace=canonical_text(scope.namespace),
            kind=scope.kind,
            id=identifier,
            parent_id=parent_path[-1] if parent_path else scope.parent_id,
            attributes=scope.attributes,
            parent_path=parent_path,
            confidence=scope.confidence,
            source=scope.source,
            inferred=scope.inferred,
        )


@dataclass(frozen=True)
class ResolvedMemoryIdentity:
    slot_id: str
    slot_uri: str
    claim_id: str
    claim_uri: str
    slot_identity: Mapping[str, Any]
    canonical_value: str
    scope_keys: tuple[str, ...]
    canonical_subject: ScopeRef
    identity_algorithm_version: str = IDENTITY_ALGORITHM_V2
    claim_identity: Mapping[str, Any] = field(default_factory=dict)
    memory_type: str = ""
    tenant_id: str = ""

    def __post_init__(self) -> None:
        if self.identity_algorithm_version != IDENTITY_ALGORITHM_V2:
            raise ValueError("resolved canonical memory identity must use Identity V2")
        if self.canonical_subject is None:
            raise ValueError("Identity V2 requires a canonical subject")
        object.__setattr__(self, "slot_identity", MappingProxyType(dict(self.slot_identity)))
        object.__setattr__(self, "claim_identity", MappingProxyType(dict(self.claim_identity)))
        object.__setattr__(self, "scope_keys", tuple(self.scope_keys))

    @property
    def canonical_subject_key(self) -> str:
        return self.canonical_subject.key


class StableMemoryIdentityResolver:
    """Identity V2: tenant + canonical subject + schema identity, never author."""

    SLOT_FIELDS = {
        "profile": ("attribute_key",),
        "preference": ("subject", "dimension"),
        "entity": ("entity_type", "canonical_entity_id"),
        "project_rule": ("rule_topic",),
        "project_decision": ("decision_topic",),
        "event": ("event_key",),
        "agent_experience": ("task_pattern", "environment_signature"),
    }

    def __init__(
        self,
        aliases: AliasRegistry | None = None,
        registry: MemoryTypeRegistry | None = None,
    ) -> None:
        self.aliases = aliases or AliasRegistry()
        self.registry = registry or MemoryTypeRegistry()

    def resolve(
        self,
        proposal: MemorySemanticProposal,
        memory_scope: MemoryScope,
        *,
        tenant_id: str,
        owner_user_id: str,
    ) -> ResolvedMemoryIdentity:
        schema = self.registry.get(proposal.memory_type)
        expected = schema.slot_identity_fields or self.SLOT_FIELDS.get(proposal.memory_type)
        if not expected:
            raise ValueError(f"no identity schema for memory type: {proposal.memory_type}")
        missing = [
            field_name
            for field_name in expected
            if proposal.identity_fields.get(field_name) is None or proposal.identity_fields.get(field_name) == ""
        ]
        if missing:
            raise ValueError(f"missing stable identity fields: {','.join(missing)}")
        slot_identity = {
            field_name: self._canonical_field(proposal.memory_type, field_name, proposal.identity_fields[field_name])
            for field_name in expected
        }
        canonical_subject = self._canonical_subject(proposal.memory_type, memory_scope)
        scopes = tuple(sorted(self.aliases.canonical_scope(scope).key for scope in memory_scope.applicability.all_of))
        slot_payload = {
            "identity_algorithm_version": IDENTITY_ALGORITHM_V2,
            "tenant_id": canonical_text(tenant_id),
            "canonical_subject": canonical_subject.key,
            "memory_type": proposal.memory_type,
            "namespace_hierarchy": canonical_subject.hierarchy_path,
            "slot_identity": slot_identity,
        }
        slot_id = stable_hash(slot_payload, length=32)
        claim_identity = self._claim_identity(schema, proposal)
        claim_id = stable_hash(
            {
                "identity_algorithm_version": IDENTITY_ALGORITHM_V2,
                "slot_id": slot_id,
                "claim_identity": claim_identity,
            },
            length=32,
        )
        canonical_value = self._canonical_value(proposal)
        storage_owner = self._storage_owner(canonical_subject, tenant_id=tenant_id, owner_user_id=owner_user_id)
        root = f"memoryos://user/{storage_owner}/memories/canonical/slots/{slot_id}"
        return ResolvedMemoryIdentity(
            slot_id=slot_id,
            slot_uri=root,
            claim_id=claim_id,
            claim_uri=f"{root}/claims/{claim_id}",
            slot_identity=slot_identity,
            canonical_value=canonical_value,
            scope_keys=scopes,
            canonical_subject=canonical_subject,
            claim_identity=claim_identity,
            memory_type=proposal.memory_type,
            tenant_id=canonical_text(tenant_id),
        )

    def _canonical_subject(self, memory_type: str, memory_scope: MemoryScope) -> ScopeRef:
        candidates = tuple(self.aliases.canonical_scope(scope) for scope in memory_scope.applicability.all_of)
        explicit = (
            self.aliases.canonical_scope(memory_scope.canonical_subject) if memory_scope.canonical_subject else None
        )
        subject: ScopeRef | None = explicit
        if subject is None:
            priorities = (
                ("principal", "workspace", "environment", "asset", "location", "global")
                if memory_type in {"profile", "preference"}
                else ("workspace", "environment", "asset", "location", "principal", "global")
                if memory_type in {"project_rule", "project_decision", "agent_experience"}
                else ("asset", "location", "workspace", "environment", "principal", "global")
            )
            subject = next((scope for kind in priorities for scope in candidates if scope.kind == kind), None)
        if subject is None:
            raise ValueError("canonical subject cannot be resolved from applicability")
        if subject.kind in HIERARCHICAL_SCOPE_KINDS and not subject.parent_path:
            container = next(
                (
                    scope
                    for kind in ("location", "environment", "workspace")
                    for scope in candidates
                    if scope.kind == kind and scope.id != subject.id
                ),
                None,
            )
            if container is not None:
                subject = replace(
                    subject,
                    parent_id=container.id,
                    parent_path=(*container.parent_path, container.key),
                    confidence=min(subject.confidence, container.confidence),
                    inferred=True,
                )
        return subject

    def _canonical_field(self, memory_type: str, field_name: str, value: Any) -> Any:
        if isinstance(value, str):
            normalized = self.aliases.resolve(f"{memory_type}:{field_name}", value)
            if field_name in {"canonical_value", "value", "decision", "rule", "preference", "name", "event", "outcome"}:
                normalized = self.aliases.resolve(f"{memory_type}:value", normalized)
            return normalized
        return canonical_identity_value(value)

    def _claim_identity(self, schema: MemoryTypeSchema, proposal: MemorySemanticProposal) -> dict[str, Any]:
        keys = schema.claim_identity_keys(dict(proposal.value_fields))
        if not keys:
            raise ValueError("claim requires schema-declared semantic value fields")
        return {
            key: self._canonical_field(proposal.memory_type, key, proposal.value_fields[key]) for key in sorted(keys)
        }

    def _canonical_value(self, proposal: MemorySemanticProposal) -> str:
        for key in ("canonical_value", "value", "decision", "rule", "preference", "name", "event", "outcome"):
            value = proposal.value_fields.get(key)
            if value is None or value == "":
                continue
            if isinstance(value, str):
                return self.aliases.resolve(f"{proposal.memory_type}:value", value)
            return canonical_identity_json(value)
        return canonical_identity_json(dict(proposal.value_fields))

    def _storage_owner(self, subject: ScopeRef, *, tenant_id: str, owner_user_id: str) -> str:
        if subject.kind == "principal" and canonical_text(subject.id) == canonical_text(owner_user_id):
            return owner_user_id
        return f"subject_{stable_hash([tenant_id, subject.key], length=20)}"
