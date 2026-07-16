"""Deterministic ordinary relation projection from Source ContextObjects."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.source_store import (
    IndexStore,
    SourceStore,
    is_canonical_memory_object,
    is_canonical_memory_uri,
)

_AUDIT_RELATION_TYPES = frozenset({"supersedes", "superseded_by"})


@dataclass(frozen=True)
class OrdinaryRelationEligibility:
    allowed: bool
    reason: str = ""


def _stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def ordinary_relation_specs_for_object(obj: ContextObject) -> list[dict[str, Any]]:
    """Return the complete ordinary relation view owned by one Source object.

    ``ContextObject.relations`` is the direct Source representation.  A small
    number of established object schemas also own relation facts in typed
    metadata; those deterministic projections remain part of the same Source
    contract so an offline rebuild does not lose ActionPolicy/Behavior edges.
    Tenant and owner always come from the authority object, never caller
    supplied relation metadata.
    """

    tenant_id = str(obj.tenant_id or "default")
    owner_user_id = str(obj.owner_user_id or "")
    authority_metadata = {
        "tenant_id": tenant_id,
        "owner_user_id": owner_user_id,
    }
    metadata = dict(obj.metadata or {})
    specs: list[dict[str, Any]] = []

    def add(
        source_uri: str,
        relation_type: str,
        target_uri: str,
        relation_metadata: dict[str, Any],
        *,
        weight: float = 1.0,
    ) -> None:
        if not source_uri or not target_uri:
            return
        normalized_metadata = dict(relation_metadata or {})
        # Ordinary relation rows are not Catalog-owned and cannot borrow an
        # immutable canonical publication identity through metadata.
        normalized_metadata.pop("catalog_record_key", None)
        normalized_metadata["tenant_id"] = tenant_id
        normalized_metadata["owner_user_id"] = owner_user_id
        specs.append(
            {
                "source_uri": str(source_uri),
                "relation_type": str(relation_type),
                "target_uri": str(target_uri),
                "weight": float(weight),
                "metadata": normalized_metadata,
            }
        )

    if obj.context_type == ContextType.ACTION_POLICY:
        add(obj.uri, "anchored_by", str(metadata.get("memory_anchor_uri", "")), authority_metadata)
        for uri in metadata.get("required_resource_uris", []) or []:
            add(obj.uri, "requires_resource", str(uri), authority_metadata)
        for uri in metadata.get("required_skill_uris", []) or []:
            add(obj.uri, "requires_skill", str(uri), authority_metadata)
        for uri in metadata.get("supported_behavior_pattern_uris", []) or []:
            add(obj.uri, "supported_by", str(uri), authority_metadata)
        for uri in metadata.get("constrained_by_memory_uris", []) or []:
            add(obj.uri, "constrained_by", str(uri), authority_metadata)
    elif obj.context_type in {ContextType.BEHAVIOR_PATTERN, ContextType.BEHAVIOR_CLUSTER}:
        add(obj.uri, "anchored_by", str(metadata.get("memory_anchor_uri", "")), authority_metadata)
        for uri in metadata.get("case_refs", []) or []:
            add(obj.uri, "aggregated_from", str(uri), authority_metadata)
        for uri in metadata.get("related_policy_uris", []) or metadata.get("policy_uris", []) or []:
            add(str(uri), "supported_by", obj.uri, authority_metadata)
    elif obj.context_type == ContextType.MEMORY:
        for policy_uri in metadata.get("constrains_policy_uris", []) or []:
            add(str(policy_uri), "constrained_by", obj.uri, authority_metadata)
        for behavior_uri in metadata.get("supporting_behavior_uris", []) or []:
            add(obj.uri, "evidence_for", str(behavior_uri), authority_metadata)

    # SUPERSEDE writes these durable Source metadata links before publishing
    # the symmetric RelationStore rows.  Rebuild them from that authority so
    # a derived clear never loses the replacement chain.
    supersedes = str(metadata.get("supersedes") or "")
    superseded_by = str(metadata.get("superseded_by") or "")
    add(obj.uri, "supersedes", supersedes, authority_metadata)
    add(supersedes, "superseded_by", obj.uri, authority_metadata)
    add(obj.uri, "superseded_by", superseded_by, authority_metadata)
    add(superseded_by, "supersedes", obj.uri, authority_metadata)

    for relation in obj.relations:
        add(
            relation.source_uri,
            relation.relation_type,
            relation.target_uri,
            dict(relation.metadata or {}),
            weight=relation.weight,
        )

    unique = {_stable_json(spec): spec for spec in specs}
    return [unique[key] for key in sorted(unique)]


def ordinary_relation_serving_eligibility(
    spec: dict[str, Any],
    *,
    authority_uri: str,
    tenant_id: str,
    source_store: SourceStore,
    index_store: IndexStore,
    authority_object: ContextObject | None = None,
    canonical_reader: Callable[[str], ContextObject] | None = None,
    allow_virtual_targets: bool = False,
) -> OrdinaryRelationEligibility:
    """Bound one ordinary edge to live Source/Catalog authority.

    Source metadata remains immutable evidence when a referenced object is
    retired.  This policy controls only the rebuildable RelationStore serving
    projection, and is shared by online publication and offline rebuild.
    """

    source_uri = str(spec.get("source_uri") or "")
    target_uri = str(spec.get("target_uri") or "")
    relation_type = str(spec.get("relation_type") or "")
    if not source_uri or not target_uri or not relation_type:
        return OrdinaryRelationEligibility(False, "relation identity is incomplete")
    metadata = dict(spec.get("metadata", {}) or {})
    declared_tenant = str(metadata.get("tenant_id") or tenant_id)
    if declared_tenant != tenant_id:
        return OrdinaryRelationEligibility(False, "relation metadata crosses its Source tenant")
    if is_canonical_memory_uri(source_uri):
        return OrdinaryRelationEligibility(False, "canonical Source requires an immutable receipt")

    audit_edge = relation_type in _AUDIT_RELATION_TYPES
    for endpoint_uri, role in ((source_uri, "source"), (target_uri, "target")):
        result = _ordinary_endpoint_eligibility(
            endpoint_uri,
            role=role,
            audit_edge=audit_edge,
            authority_uri=authority_uri,
            tenant_id=tenant_id,
            source_store=source_store,
            index_store=index_store,
            authority_object=authority_object,
            canonical_reader=canonical_reader,
            allow_virtual_targets=allow_virtual_targets,
        )
        if not result.allowed:
            return result
    return OrdinaryRelationEligibility(True)


def _ordinary_endpoint_eligibility(
    uri: str,
    *,
    role: str,
    audit_edge: bool,
    authority_uri: str,
    tenant_id: str,
    source_store: SourceStore,
    index_store: IndexStore,
    authority_object: ContextObject | None,
    canonical_reader: Callable[[str], ContextObject] | None,
    allow_virtual_targets: bool,
) -> OrdinaryRelationEligibility:
    if not uri.startswith("memoryos://"):
        # External evidence nodes have no local lifecycle.  Their containing
        # Source authority still gates publication.
        return OrdinaryRelationEligibility(True)
    try:
        parsed = ContextURI.parse(uri)
    except (TypeError, ValueError):
        return OrdinaryRelationEligibility(False, f"{role} URI is invalid")

    if is_canonical_memory_uri(uri):
        if role == "source":
            return OrdinaryRelationEligibility(False, "canonical Source requires an immutable receipt")
        if canonical_reader is None:
            return OrdinaryRelationEligibility(False, "canonical committed-state reader is unavailable")
        # The callback proves Source bundle + receipt + Current Head.  Its
        # integrity errors deliberately propagate so rebuild cannot disguise
        # canonical corruption as an absent ordinary edge.
        committed = canonical_reader(uri)
        return _canonical_target_eligibility(
            committed,
            role=role,
            source_store=source_store,
            tenant_id=tenant_id,
            canonical_reader=canonical_reader,
        )

    try:
        obj = source_store.read_object(uri)
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        obj = None
    if obj is None and authority_object is not None and uri == authority_uri:
        obj = authority_object

    session_id = _session_id(parsed)
    endpoint_tenant = (
        str(obj.tenant_id or "default")
        if obj is not None and parsed.authority in {"resources", "skills"}
        else tenant_id
    )
    endpoint_state = getattr(index_store, "ordinary_relation_endpoint_state", None)
    if callable(endpoint_state):
        state = str(endpoint_state(uri, tenant_id=endpoint_tenant, session_id=session_id) or "missing")
        if state == "retired":
            return OrdinaryRelationEligibility(False, f"{role} endpoint is retired")
    elif session_id:
        return OrdinaryRelationEligibility(False, f"{role} Session lifecycle is unavailable")
    else:
        state = "missing"

    if obj is not None:
        if parsed.authority == "user" and str(obj.tenant_id or "default") != tenant_id:
            return OrdinaryRelationEligibility(False, f"{role} endpoint crosses its Source tenant")
        allowed_lifecycle = {LifecycleState.ACTIVE}
        if audit_edge:
            allowed_lifecycle.add(LifecycleState.OBSOLETE)
        if obj.lifecycle_state not in allowed_lifecycle:
            return OrdinaryRelationEligibility(False, f"{role} endpoint is not serving")
        if is_canonical_memory_object(obj):
            if not is_canonical_memory_uri(uri):
                return OrdinaryRelationEligibility(False, "canonical endpoint URI is invalid")
            return _canonical_target_eligibility(
                obj,
                role=role,
                source_store=source_store,
                tenant_id=tenant_id,
                canonical_reader=canonical_reader,
            )
        return OrdinaryRelationEligibility(True)

    if is_canonical_memory_uri(uri):
        return OrdinaryRelationEligibility(False, f"{role} canonical endpoint is missing")
    if session_id:
        if state != "active":
            return OrdinaryRelationEligibility(False, f"{role} Session endpoint has no active Catalog row")
        return OrdinaryRelationEligibility(True)
    if parsed.authority in {"resources", "skills"}:
        # Global registries can be served outside the tenant-bound SourceStore.
        return OrdinaryRelationEligibility(True)
    if uri == authority_uri:
        return OrdinaryRelationEligibility(False, "relation Source authority is missing")
    # Compatibility: a durable target authority may own an edge whose logical
    # source is an external/legacy user node that never had a Source object.
    if role == "source":
        return OrdinaryRelationEligibility(True)
    if allow_virtual_targets:
        return OrdinaryRelationEligibility(True)
    return OrdinaryRelationEligibility(False, "ordinary target Source is missing")


def _canonical_target_eligibility(
    obj: ContextObject,
    *,
    role: str,
    source_store: SourceStore,
    tenant_id: str,
    canonical_reader: Callable[[str], ContextObject] | None,
) -> OrdinaryRelationEligibility:
    if role == "source":
        return OrdinaryRelationEligibility(False, "canonical Source requires an immutable receipt")
    metadata = dict(obj.metadata or {})
    kind = str(metadata.get("canonical_kind") or "")
    if kind == "claim":
        try:
            current_revision = int(metadata.get("current_revision", metadata.get("revision", 0)) or 0)
        except (TypeError, ValueError):
            return OrdinaryRelationEligibility(False, "canonical Claim current revision is invalid")
        revisions = [
            dict(item)
            for item in metadata.get("revisions", []) or []
            if isinstance(item, dict) and int(item.get("revision", 0) or 0) == current_revision
        ]
        if len(revisions) != 1 or str(revisions[0].get("state") or "").upper() != "ACTIVE":
            return OrdinaryRelationEligibility(False, "canonical Claim is not ACTIVE")
        if str(metadata.get("state") or "").upper() != "ACTIVE":
            return OrdinaryRelationEligibility(False, "canonical Claim materialized state is inconsistent")
        slot_uri = obj.uri.rsplit("/claims/", 1)[0]
        try:
            slot = canonical_reader(slot_uri) if canonical_reader is not None else source_store.read_object(slot_uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return OrdinaryRelationEligibility(False, "canonical Claim Slot is missing")
        slot_metadata = dict(slot.metadata or {})
        if (
            str(slot.tenant_id or "default") != tenant_id
            or str(slot_metadata.get("canonical_kind") or "") != "slot"
            or str(slot_metadata.get("active_claim_id") or "") != str(metadata.get("claim_id") or "")
        ):
            return OrdinaryRelationEligibility(False, "canonical Claim is not its Slot current state")
        return OrdinaryRelationEligibility(True)
    if kind == "slot":
        if not str(metadata.get("active_claim_id") or ""):
            return OrdinaryRelationEligibility(False, "canonical Slot has no ACTIVE Claim")
        return OrdinaryRelationEligibility(True)
    return OrdinaryRelationEligibility(False, "canonical endpoint is not a serving Slot or Claim")


def _session_id(uri: ContextURI) -> str:
    segments = uri.segments
    for index in range(len(segments) - 2):
        if segments[index : index + 2] == ("sessions", "history"):
            return str(segments[index + 2])
    return ""


__all__ = [
    "OrdinaryRelationEligibility",
    "ordinary_relation_serving_eligibility",
    "ordinary_relation_specs_for_object",
]
