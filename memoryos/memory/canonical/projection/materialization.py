"""Materialization responsibilities for canonical projection."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from memoryos.contextdb.catalog import (
    CatalogProjectionStatus,
    CatalogRecordKind,
    ServingTier,
)
from memoryos.contextdb.model.context_layer import ContextLayers
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.memory.canonical.projection_state import (
    ProjectionIntegrityError,
    ProjectionRecord,
)
from memoryos.memory.canonical.state import (
    materialized_current_revision_payload,
    revision_payload_with_effective_validity,
)

from .models import (
    _MAX_CLAIM_REVISION_REFRESH,
)

if TYPE_CHECKING:
    from .service import CanonicalMemoryProjector


def _layers(
    self: CanonicalMemoryProjector,
    obj: ContextObject,
    metadata: dict[str, Any],
    revision: dict[str, Any],
    source_revision: int,
) -> tuple[str, str, str]:
    revision_values = dict(revision.get("value_fields", {}) or {})
    value = str(
        revision_values.get("canonical_value")
        or revision_values.get("value")
        or metadata.get("canonical_value", obj.title)
    )
    state = str(revision.get("state") or metadata.get("state", ""))
    memory_type = str(metadata.get("memory_type", "memory"))
    l0 = f"{value} [{state}]"
    qualifiers = dict(revision.get("qualifiers", {}) or {})
    display_fields = dict(qualifiers.get("display_fields", {}) or {})
    display_field_evidence_refs = dict(qualifiers.get("display_field_evidence_refs", {}) or {})
    l1_lines = [
        f"# {value}",
        f"- type: {memory_type}",
        f"- state: {state}",
        f"- source revision: {source_revision}",
        f"- current claim revision: {revision.get('revision', source_revision)}",
        f"- epistemic: {revision.get('epistemic_status', '')}",
        f"- relation: {revision.get('relation', '')}",
    ]
    display_text = next(
        (
            str(display_fields[name])
            for name in ("display_text", "summary", "decision", "rule", "rationale", "details", "reason")
            if display_fields.get(name)
        ),
        "",
    )
    if display_text:
        l1_lines.append(f"- display: {display_text}")
    if qualifiers:
        l1_lines.append(f"- qualifiers: {json.dumps(qualifiers, ensure_ascii=False, sort_keys=True)}")
    l1 = "\n".join(l1_lines)
    l2 = json.dumps(
        {
            "claim_uri": obj.uri,
            "slot_id": metadata.get("slot_id"),
            "claim_id": metadata.get("claim_id"),
            "source_revision": source_revision,
            "current_claim_revision": revision.get("revision", source_revision),
            "canonical_value": value,
            "revision": revision,
            "display_fields": display_fields,
            "display_field_evidence_refs": display_field_evidence_refs,
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return l0, l1, l2


def _sanitized_revision_layers(
    self: CanonicalMemoryProjector,
    obj: ContextObject,
    metadata: dict[str, Any],
    revision: dict[str, Any],
    source_revision: int,
) -> tuple[str, str, str]:
    l0, l1, l2 = self._layers(obj, metadata, revision, source_revision)
    safe = self.sanitizer.sanitize(
        title=obj.title,
        l0_text=l0,
        l1_text=l1,
        metadata={"l2": json.loads(l2)},
        source_kind="canonical_claim",
    )
    l2_payload = safe.metadata.get("l2")
    if not isinstance(l2_payload, dict):
        raise ProjectionIntegrityError("canonical revision L2 sanitization returned an invalid payload")
    return (
        safe.l0_text,
        safe.l1_text,
        json.dumps(l2_payload, ensure_ascii=False, indent=2, sort_keys=True),
    )


def _bounded_claim_revisions(metadata: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_revisions = metadata.get("revisions", ()) or ()
    if not isinstance(raw_revisions, (list, tuple)):
        raise ProjectionIntegrityError("canonical Claim revisions are not an array")
    if not raw_revisions or len(raw_revisions) > _MAX_CLAIM_REVISION_REFRESH:
        raise ProjectionIntegrityError("canonical Claim revision refresh exceeds its bounded limit")
    if any(not isinstance(item, dict) for item in raw_revisions):
        raise ProjectionIntegrityError("canonical Claim revision payload is invalid")
    return tuple(dict(item) for item in raw_revisions)


def _revision_payload(self: CanonicalMemoryProjector, metadata: dict[str, Any], revision: int) -> dict[str, Any]:
    revisions = [dict(item) for item in metadata.get("revisions", []) or [] if int(item.get("revision", 0)) == revision]
    if not revisions:
        raise ValueError("canonical claim revision payload is missing")
    return revisions[-1]


def _projection_object(
    self: CanonicalMemoryProjector,
    obj: ContextObject,
    metadata: dict[str, Any],
    record: ProjectionRecord,
    *,
    domain_identity: dict[str, Any],
    layers: ContextLayers,
) -> ContextObject:
    projected = ContextObject.from_dict(obj.to_dict())
    projected.layers = layers
    materialized_current = materialized_current_revision_payload(metadata)
    current_revision = revision_payload_with_effective_validity(
        tuple(metadata.get("revisions", ()) or ()),
        int(materialized_current["revision"]),
    )
    tree_paths = self._canonical_tree_paths(metadata)
    source_timestamp = str(
        current_revision.get("transaction_time")
        or current_revision.get("created_at")
        or projected.updated_at
        or projected.created_at
    )
    valid_from = str(current_revision.get("valid_from") or "")
    valid_to = str(current_revision.get("valid_to") or "")
    safe = self.sanitizer.sanitize(
        title=projected.title,
        metadata={
            **metadata,
            **domain_identity,
            "record_kind": CatalogRecordKind.CLAIM_REVISION.value,
            "source_kind": "canonical_claim",
            "catalog_record_key": self._claim_catalog_record_key(metadata, record.source_revision),
            "tree_paths": list(tree_paths),
            "primary_tree_path": tree_paths[0],
            "source_uri": projected.uri,
            "source_digest": record.projected_content_digest,
            "source_revision": record.source_revision,
            "event_time": str(current_revision.get("event_time") or valid_from or projected.created_at),
            "ingested_at": str(current_revision.get("created_at") or projected.created_at),
            "transaction_time": source_timestamp,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "serving_tier": ServingTier.HOT.value,
            "projection_status": CatalogProjectionStatus.PROJECTED.value,
            "projection_effect_hash": record.input_effect_hash,
            "projection_source_revision": record.source_revision,
            "projection_revision": record.projection_revision,
            "projection_attempt_id": record.projection_attempt_id,
            "projection_input_effect_hash": record.input_effect_hash,
            "projection_publish_token": record.publish_token,
            "projection_content_digest": record.projected_content_digest,
            "projection_relation_digest": record.projected_relation_digest,
            "current_claim_revision": record.current_claim_revision,
            "projection_manifest_uri": record.manifest_uri,
            "projection_record_path": str(self.record_store.attempt_path_for(record)),
        },
        source_kind="canonical_claim",
    )
    projected.title = safe.title
    projected.metadata = safe.metadata
    return projected


def _manifest(
    self: CanonicalMemoryProjector,
    record: ProjectionRecord,
    metadata: dict[str, Any],
    relations_uri: str,
    *,
    domain_identity: dict[str, Any],
) -> dict[str, Any]:
    return {
        **record.to_dict(),
        **domain_identity,
        "memory_id": metadata.get("claim_id"),
        "slot_id": metadata.get("slot_id"),
        "claim_id": metadata.get("claim_id"),
        "projection_levels": ["L0", "L1", "L2"],
        "projections": [
            {
                "claim_uri": record.claim_uri,
                "slot_uri": record.slot_uri,
                "source_revision": record.source_revision,
                "projection_revision": record.projection_revision,
                "projection_attempt_id": record.projection_attempt_id,
                "input_effect_hash": record.input_effect_hash,
                "publish_token": record.publish_token,
                "projection_level": level,
                "uri": uri,
                "generator": self.GENERATOR,
                "model_id": None,
                "prompt_version": self.PROMPT_VERSION,
                "created_at": record.created_at,
            }
            for level, uri in (("L0", record.l0_uri), ("L1", record.l1_uri), ("L2", record.l2_uri))
        ],
        "relation_projection_uri": relations_uri,
        "generator": self.GENERATOR,
        "model_id": None,
        "prompt_version": self.PROMPT_VERSION,
    }
