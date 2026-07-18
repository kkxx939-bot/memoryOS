"""Durable, verifiable proofs for committed SourceStore effects."""

from __future__ import annotations

import json
import os as os
from pathlib import Path
from typing import Any

from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.relation_store import RelationStore
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.core.clock import utc_now
from memoryos.core.durable_io.atomic_file import (
    ImmutableArtifactConflictError as ImmutableArtifactConflictError,
)
from memoryos.core.durable_io.atomic_file import atomic_create_bytes as atomic_create_bytes
from memoryos.core.durable_io.atomic_json import atomic_create_json as atomic_create_json
from memoryos.core.durable_io.atomic_json import atomic_write_json as atomic_write_json
from memoryos.core.integrity import canonical_digest, canonical_json

EFFECT_MARKER_SCHEMA_VERSION = "effect_marker_v1"


class EffectProofError(RuntimeError):
    """A durable marker cannot prove the SourceStore effects it claims."""


def relation_identity(spec: dict[str, Any]) -> dict[str, str]:
    return {
        "source_uri": str(spec.get("source_uri") or ""),
        "relation_type": str(spec.get("relation_type") or spec.get("type") or ""),
        "target_uri": str(spec.get("target_uri") or ""),
    }


def normalized_relation(value: ContextRelation | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, ContextRelation):
        return {
            **relation_identity(value.to_dict()),
            "weight": float(value.weight),
            "metadata": dict(value.metadata or {}),
        }
    return {
        **relation_identity(value),
        "weight": float(value.get("weight", 1.0)),
        "metadata": dict(value.get("metadata", {}) or {}),
    }


def relation_effects_from_manifest(manifest: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict):
        return []
    effects: list[dict[str, Any]] = []
    for item in manifest.get("expected", []) or []:
        if not isinstance(item, dict):
            raise EffectProofError("relation manifest contains a non-object expected effect")
        relation = normalized_relation(item)
        effects.append(
            {
                "identity": relation_identity(relation),
                **relation_identity(relation),
                "expected_exists": True,
                "relation_digest": canonical_digest(relation),
                "relation": relation,
            }
        )
    expected_identities = {canonical_json(relation_identity(effect["identity"])) for effect in effects}
    for item in manifest.get("remove", []) or []:
        if not isinstance(item, dict):
            raise EffectProofError("relation manifest contains a non-object removal effect")
        identity = relation_identity(item)
        if canonical_json(identity) in expected_identities:
            continue
        effects.append(
            {
                "identity": identity,
                **identity,
                "expected_exists": False,
                "relation_digest": canonical_digest(identity),
            }
        )
    unique = {
        canonical_json(
            {
                "identity": effect["identity"],
                "expected_exists": effect["expected_exists"],
            }
        ): effect
        for effect in effects
    }
    return [unique[key] for key in sorted(unique)]


def object_effect_from_store(
    source_store: SourceStore,
    uri: str,
    *,
    operation_type: str,
    expected_exists: bool = True,
    logical_absence: bool = False,
) -> dict[str, Any]:
    try:
        obj = source_store.read_object(uri)
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        if expected_exists or logical_absence:
            raise EffectProofError(f"committed object effect is missing: {uri}") from None
        return {
            "uri": uri,
            "expected_exists": False,
            "absence_mode": "physical",
            "object_digest": canonical_digest(None),
            "content_digest": canonical_digest(None),
            "content_uri": "",
            "revision": None,
            "operation_type": operation_type,
        }
    if not expected_exists and not logical_absence:
        raise EffectProofError(f"committed object effect unexpectedly exists: {uri}")
    if logical_absence and obj.lifecycle_state != LifecycleState.DELETED:
        raise EffectProofError(f"committed logical deletion is not a tombstone: {uri}")
    content_uri = str(obj.layers.l2_uri or obj.uri)
    try:
        content: str | None = source_store.read_content(content_uri)
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        content = None
    metadata = dict(obj.metadata or {})
    return {
        "uri": uri,
        "expected_exists": expected_exists,
        "absence_mode": "logical_deleted" if logical_absence else "present",
        "object_digest": canonical_digest(obj.to_dict()),
        "content_digest": canonical_digest(content),
        "content_uri": content_uri,
        "revision": metadata.get("revision"),
        "operation_type": operation_type,
        "tenant_id": str(obj.tenant_id or "default"),
        "user_id": str(obj.owner_user_id or ""),
    }


def build_marker(
    *,
    transaction_id: str,
    idempotency_key: str,
    tenant_id: str,
    user_id: str,
    operation_ids: list[str],
    object_effects: list[dict[str, Any]],
    relation_effects: list[dict[str, Any]],
    diff: dict[str, Any],
    operations: list[dict[str, Any]],
) -> dict[str, Any]:
    core: dict[str, Any] = {
        "schema_version": EFFECT_MARKER_SCHEMA_VERSION,
        "status": "committed",
        "transaction_id": transaction_id,
        "idempotency_key": idempotency_key,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "committed_at": utc_now(),
        "operation_ids": list(operation_ids),
        "object_effects": list(object_effects),
        "relation_effects": list(relation_effects),
        "diff": diff,
        "operations": operations,
    }
    return {**core, "marker_digest": canonical_digest(core)}


def load_marker(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise EffectProofError("transaction marker cannot be a symbolic link")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EffectProofError("transaction marker is unreadable") from exc
    if not isinstance(payload, dict):
        raise EffectProofError("transaction marker must be a JSON object")
    return payload


def validate_marker(
    path: Path,
    source_store: SourceStore,
    relation_store: RelationStore | None = None,
    *,
    transaction_id: str | None = None,
    idempotency_key: str | None = None,
    tenant_id: str | None = None,
    user_id: str | None = None,
    operation_ids: list[str] | None = None,
    object_uris: set[str] | None = None,
) -> dict[str, Any]:
    payload = load_marker(path)
    if payload.get("schema_version") != EFFECT_MARKER_SCHEMA_VERSION:
        raise EffectProofError("transaction marker schema is unsupported")
    if payload.get("status") != "committed":
        raise EffectProofError("transaction marker is not committed")
    digest = payload.get("marker_digest")
    core = {key: value for key, value in payload.items() if key != "marker_digest"}
    if not isinstance(digest, str) or digest != canonical_digest(core):
        raise EffectProofError("transaction marker digest is corrupt")
    expected_bindings: tuple[tuple[str, object | None], ...] = (
        ("transaction_id", transaction_id),
        ("idempotency_key", idempotency_key),
        ("tenant_id", tenant_id),
        ("user_id", user_id),
    )
    for key, expected in expected_bindings:
        if expected is not None and payload.get(key) != expected:
            raise EffectProofError(f"transaction marker {key} does not match")
    if operation_ids is not None and list(payload.get("operation_ids", []) or []) != list(operation_ids):
        raise EffectProofError("transaction marker operation ids do not match")
    object_effects = payload.get("object_effects")
    relation_effects = payload.get("relation_effects")
    if not isinstance(object_effects, list) or not object_effects:
        raise EffectProofError("transaction marker has no object effect proof")
    if not isinstance(relation_effects, list):
        raise EffectProofError("transaction marker relation effects are invalid")
    selected_object_effects = [
        effect
        for effect in object_effects
        if object_uris is None or (isinstance(effect, dict) and str(effect.get("uri") or "") in object_uris)
    ]
    if (
        object_uris is not None
        and {str(effect.get("uri") or "") for effect in selected_object_effects if isinstance(effect, dict)}
        != object_uris
    ):
        raise EffectProofError("transaction marker does not prove the requested object effects")
    for effect in selected_object_effects:
        _validate_object_effect(
            source_store,
            effect,
            tenant_id=str(payload["tenant_id"]),
            user_id=str(payload["user_id"]),
        )
    if relation_store is not None:
        for effect in relation_effects:
            _validate_relation_effect(
                relation_store,
                effect,
                tenant_id=str(payload["tenant_id"]),
            )
    return payload


def _validate_object_effect(
    source_store: SourceStore,
    effect: object,
    *,
    tenant_id: str,
    user_id: str,
) -> None:
    if not isinstance(effect, dict) or not isinstance(effect.get("uri"), str):
        raise EffectProofError("transaction marker object effect is invalid")
    uri = str(effect["uri"])
    expected_exists = effect.get("expected_exists")
    if not isinstance(expected_exists, bool):
        raise EffectProofError("transaction marker object existence proof is invalid")
    absence_mode = str(effect.get("absence_mode") or "")
    try:
        obj = source_store.read_object(uri)
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        if expected_exists or absence_mode != "physical":
            raise EffectProofError(f"transaction marker object effect is missing: {uri}") from None
        if effect.get("object_digest") != canonical_digest(None) or effect.get("content_digest") != canonical_digest(
            None
        ):
            raise EffectProofError("physical deletion proof digest is corrupt") from None
        return
    if not expected_exists:
        if absence_mode != "logical_deleted" or obj.lifecycle_state != LifecycleState.DELETED:
            raise EffectProofError(f"transaction marker expected object absence: {uri}")
    elif absence_mode != "present":
        raise EffectProofError("transaction marker object presence mode is invalid")
    if str(obj.tenant_id or "default") != tenant_id:
        raise EffectProofError("transaction marker object crosses a tenant boundary")
    if str(obj.owner_user_id or "") not in {"", user_id}:
        raise EffectProofError("transaction marker object crosses a user boundary")
    if effect.get("object_digest") != canonical_digest(obj.to_dict()):
        raise EffectProofError(f"transaction marker object digest does not match: {uri}")
    metadata = dict(obj.metadata or {})
    if effect.get("revision") != metadata.get("revision"):
        raise EffectProofError(f"transaction marker object revision does not match: {uri}")
    content_uri = str(effect.get("content_uri") or obj.layers.l2_uri or obj.uri)
    try:
        content: str | None = source_store.read_content(content_uri)
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        content = None
    if effect.get("content_digest") != canonical_digest(content):
        raise EffectProofError(f"transaction marker content digest does not match: {uri}")


def _validate_relation_effect(
    relation_store: RelationStore,
    effect: object,
    *,
    tenant_id: str,
) -> None:
    if not isinstance(effect, dict) or not isinstance(effect.get("identity"), dict):
        raise EffectProofError("transaction marker relation effect is invalid")
    identity = relation_identity(effect["identity"])
    if not all(identity.values()) or not isinstance(effect.get("expected_exists"), bool):
        raise EffectProofError("transaction marker relation identity is invalid")
    matches = [
        relation
        for relation in relation_store.relations_of(
            identity["source_uri"],
            tenant_id=tenant_id,
        )
        if relation.source_uri == identity["source_uri"]
        and relation.relation_type == identity["relation_type"]
        and relation.target_uri == identity["target_uri"]
    ]
    if effect["expected_exists"]:
        if len(matches) != 1:
            raise EffectProofError("transaction marker relation effect is missing or ambiguous")
        if effect.get("relation_digest") != canonical_digest(normalized_relation(matches[0])):
            raise EffectProofError("transaction marker relation digest does not match")
    elif matches or effect.get("relation_digest") != canonical_digest(identity):
        raise EffectProofError("transaction marker relation absence proof does not match")


def marker_proves_object(payload: dict[str, Any], uri: str) -> bool:
    return any(
        isinstance(effect, dict) and effect.get("uri") == uri and effect.get("expected_exists") is True
        for effect in payload.get("object_effects", []) or []
    )


def marker_proves_relation(payload: dict[str, Any], relation: ContextRelation) -> bool:
    desired = normalized_relation(relation)
    desired_identity = relation_identity(desired)
    desired_digest = canonical_digest(desired)
    return any(
        isinstance(effect, dict)
        and effect.get("expected_exists") is True
        and effect.get("identity") == desired_identity
        and effect.get("relation_digest") == desired_digest
        for effect in payload.get("relation_effects", []) or []
    )
