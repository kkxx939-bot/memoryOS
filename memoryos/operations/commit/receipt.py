"""Immutable receipts for canonical and pending-memory transactions.

A receipt proves what one transaction committed.  It deliberately never reads
the current SourceStore: later revisions must not invalidate historical proof.
Current-state proof lives in ``memoryos.memory.canonical.current_head``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from memoryos.core.time import utc_now
from memoryos.memory.canonical.event import canonical_digest, canonical_json

TRANSACTION_RECEIPT_SCHEMA_VERSION = "memory_transaction_receipt_v2"


class ReceiptIntegrityError(RuntimeError):
    """An immutable receipt is corrupt or crosses its declared boundary."""


def _normalized_operation(operation: Any) -> dict[str, Any]:
    payload = operation.to_dict() if hasattr(operation, "to_dict") else dict(operation)
    payload.pop("status", None)
    return payload


def operation_set_digest(operations: Sequence[Any]) -> str:
    normalized = sorted(
        (_normalized_operation(operation) for operation in operations),
        key=lambda item: str(item.get("operation_id") or ""),
    )
    return canonical_digest(normalized)


def effect_snapshots(
    operations: Sequence[Any],
    *,
    relation_effects: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Materialize immutable object/content/relation snapshots from intent."""

    for effect in relation_effects:
        if not isinstance(effect, dict):
            raise ReceiptIntegrityError("receipt relation effect must be an object")

    snapshots: list[dict[str, Any]] = []
    for operation in sorted(operations, key=lambda item: str(_normalized_operation(item).get("operation_id") or "")):
        normalized = _normalized_operation(operation)
        payload = dict(normalized.get("payload", {}) or {})
        object_payload = payload.get("context_object")
        if not isinstance(object_payload, dict) or not object_payload.get("uri"):
            raise ReceiptIntegrityError("receipt operation is missing its object snapshot")
        uri = str(object_payload["uri"])
        content = str(payload.get("content", ""))
        # A revision proves the complete current bundle, not merely this
        # transaction's relation delta. The separate relation_effects member
        # still binds the immutable add/remove intent.
        raw_relations = object_payload.get("relations", [])
        if not isinstance(raw_relations, list):
            raise ReceiptIntegrityError("receipt object relations must be a list")
        relations = sorted((dict(item) for item in raw_relations if isinstance(item, dict)), key=canonical_json)
        if len(relations) != len(raw_relations):
            raise ReceiptIntegrityError("receipt object relations contain an invalid member")
        relation_snapshot = sorted(
            (
                {
                    "source_uri": str(dict(effect.get("relation", {}) or {}).get("source_uri") or ""),
                    "relation_type": str(
                        dict(effect.get("relation", {}) or {}).get("relation_type")
                        or dict(effect.get("relation", {}) or {}).get("type")
                        or ""
                    ),
                    "target_uri": str(dict(effect.get("relation", {}) or {}).get("target_uri") or ""),
                    "weight": float(dict(effect.get("relation", {}) or {}).get("weight", 1.0)),
                    "metadata": dict(dict(effect.get("relation", {}) or {}).get("metadata", {}) or {}),
                }
                for effect in relation_effects
                if isinstance(effect, dict)
                and effect.get("expected_exists") is True
                and isinstance(effect.get("relation"), dict)
                and str(dict(effect.get("identity", {}) or {}).get("source_uri") or "") == uri
            ),
            key=canonical_json,
        )
        metadata = dict(object_payload.get("metadata", {}) or {})
        core: dict[str, Any] = {
            "operation_id": str(normalized.get("operation_id") or ""),
            "uri": uri,
            "canonical_kind": str(metadata.get("canonical_kind") or ""),
            "expected_exists": True,
            "object": object_payload,
            "content": content,
            "relations": relations,
            # ``relations`` is the relation representation stored inside the
            # atomic Source bundle.  ``relation_snapshot`` is the complete
            # formal outgoing RelationStore state proved by this transaction.
            # Keeping both digests prevents either representation being
            # mistaken for the other during recovery.
            "relation_snapshot": relation_snapshot,
            "object_digest": canonical_digest(object_payload),
            "content_digest": canonical_digest(content),
            "bundle_relation_digest": canonical_digest(relations),
            "relation_digest": canonical_digest(relation_snapshot),
            "before_revision": payload.get(
                "expected_pending_lifecycle_revision",
                payload.get("expected_revision", 0),
            ),
            "after_revision": metadata.get(
                "lifecycle_revision",
                metadata.get("revision", 0),
            ),
        }
        snapshots.append({**core, "effect_digest": canonical_digest(core)})
    return snapshots


def build_transaction_receipt(
    *,
    transaction_id: str,
    idempotency_key: str,
    tenant_id: str,
    user_id: str,
    commit_group_id: str,
    operations: Sequence[Any],
    diff: dict[str, Any],
    planning_digest: str,
    prepared_intent_digest: str,
    prepared_intent_schema_version: str = "",
    relation_effects: Sequence[dict[str, Any]] = (),
    created_at: str = "",
) -> dict[str, Any]:
    normalized = [_normalized_operation(operation) for operation in operations]
    if not normalized:
        raise ReceiptIntegrityError("transaction receipt requires operations")
    operation_ids = [str(operation.get("operation_id") or "") for operation in normalized]
    if not all(operation_ids) or len(operation_ids) != len(set(operation_ids)):
        raise ReceiptIntegrityError("transaction receipt operation ids are invalid")
    if not planning_digest or not prepared_intent_digest:
        raise ReceiptIntegrityError("transaction receipt requires planning and prepared-intent digests")
    snapshots = effect_snapshots(normalized, relation_effects=relation_effects)
    before_revisions = {str(item["uri"]): item["before_revision"] for item in snapshots}
    after_revisions = {str(item["uri"]): item["after_revision"] for item in snapshots}
    core: dict[str, Any] = {
        "schema_version": TRANSACTION_RECEIPT_SCHEMA_VERSION,
        "transaction_id": transaction_id,
        "operation_id": operation_ids[0] if len(operation_ids) == 1 else "",
        "idempotency_key": idempotency_key,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "commit_group_id": commit_group_id,
        "operation_ids": operation_ids,
        "operation_set_digest": operation_set_digest(normalized),
        "planning_digest": planning_digest,
        "prepared_intent_digest": prepared_intent_digest,
        **(
            {"prepared_intent_schema_version": prepared_intent_schema_version} if prepared_intent_schema_version else {}
        ),
        "diff_digest": canonical_digest(diff),
        "planned_effect_digests": [str(item["effect_digest"]) for item in snapshots],
        "before_revisions": before_revisions,
        "after_revisions": after_revisions,
        "effect_snapshots": snapshots,
        "relation_effects": [dict(item) for item in relation_effects],
        "diff": diff,
        "operations": normalized,
        "created_at": created_at or utc_now(),
    }
    receipt = {**core, "receipt_digest": canonical_digest(core)}
    return validate_transaction_receipt(receipt)


def load_transaction_receipt(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ReceiptIntegrityError("immutable transaction receipt cannot be a symbolic link")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReceiptIntegrityError("transaction receipt is unreadable") from exc
    return validate_transaction_receipt(payload)


def validate_transaction_receipt(
    payload: object,
    *,
    transaction_id: str | None = None,
    idempotency_key: str | None = None,
    tenant_id: str | None = None,
    user_id: str | None = None,
    commit_group_id: str | None = None,
    operation_ids: Sequence[str] | None = None,
    planning_digest: str | None = None,
    prepared_intent_digest: str | None = None,
) -> dict[str, Any]:
    """Validate immutable identity and snapshots without consulting live state."""

    if not isinstance(payload, dict):
        raise ReceiptIntegrityError("transaction receipt must be a JSON object")
    if payload.get("schema_version") != TRANSACTION_RECEIPT_SCHEMA_VERSION:
        raise ReceiptIntegrityError("transaction receipt schema is unsupported")
    digest = payload.get("receipt_digest")
    core = {key: value for key, value in payload.items() if key != "receipt_digest"}
    if not isinstance(digest, str) or digest != canonical_digest(core):
        raise ReceiptIntegrityError("transaction receipt digest is corrupt")
    for key, expected in (
        ("transaction_id", transaction_id),
        ("idempotency_key", idempotency_key),
        ("tenant_id", tenant_id),
        ("user_id", user_id),
        ("commit_group_id", commit_group_id),
        ("planning_digest", planning_digest),
        ("prepared_intent_digest", prepared_intent_digest),
    ):
        if expected is not None and payload.get(key) != expected:
            raise ReceiptIntegrityError(f"transaction receipt {key} does not match")
    stored_operations = payload.get("operations")
    stored_ids = payload.get("operation_ids")
    snapshots = payload.get("effect_snapshots")
    relation_effects = payload.get("relation_effects")
    diff = payload.get("diff")
    if (
        not isinstance(stored_operations, list)
        or not stored_operations
        or not isinstance(stored_ids, list)
        or not isinstance(snapshots, list)
        or not isinstance(relation_effects, list)
        or not isinstance(diff, dict)
    ):
        raise ReceiptIntegrityError("transaction receipt members are incomplete")
    actual_ids = [str(item.get("operation_id") or "") for item in stored_operations if isinstance(item, dict)]
    if (
        actual_ids != [str(item) for item in stored_ids]
        or len(actual_ids) != len(stored_operations)
        or not all(actual_ids)
        or len(actual_ids) != len(set(actual_ids))
    ):
        raise ReceiptIntegrityError("transaction receipt operation ids are inconsistent")
    if operation_ids is not None and list(operation_ids) != actual_ids:
        raise ReceiptIntegrityError("transaction receipt operation ids do not match")
    if payload.get("operation_set_digest") != operation_set_digest(stored_operations):
        raise ReceiptIntegrityError("transaction receipt operation set is corrupt")
    if payload.get("diff_digest") != canonical_digest(diff):
        raise ReceiptIntegrityError("transaction receipt diff is corrupt")
    diff_operations = diff.get("operations")
    if (
        diff.get("user_id") != payload.get("user_id")
        or not isinstance(diff.get("diff_id"), str)
        or not diff.get("diff_id")
        or not isinstance(diff_operations, list)
        or operation_set_digest(diff_operations) != payload.get("operation_set_digest")
    ):
        raise ReceiptIntegrityError("transaction receipt diff is not bound to its operation set")
    transaction_id_value = str(payload.get("transaction_id") or "")
    idempotency_value = str(payload.get("idempotency_key") or "")
    tenant_value = str(payload.get("tenant_id") or "")
    user_value = str(payload.get("user_id") or "")
    commit_group_value = payload.get("commit_group_id")
    if not all((transaction_id_value, idempotency_value, tenant_value, user_value)):
        raise ReceiptIntegrityError("transaction receipt identity is incomplete")
    if not isinstance(commit_group_value, str) or not commit_group_value:
        raise ReceiptIntegrityError("transaction receipt commit group identity is incomplete")
    for operation in stored_operations:
        if not isinstance(operation, dict):
            raise ReceiptIntegrityError("transaction receipt operation is invalid")
        operation_payload = operation.get("payload")
        if (
            operation.get("user_id") != user_value
            or not isinstance(operation_payload, dict)
            or str(operation_payload.get("transaction_id") or "") != transaction_id_value
            or str(operation_payload.get("idempotency_key") or "") != idempotency_value
            or str(operation_payload.get("tenant_id") or "") != tenant_value
            or str(operation_payload.get("commit_group_id") or "") != commit_group_value
            or str(operation_payload.get("planning_digest") or "") != payload.get("planning_digest")
        ):
            raise ReceiptIntegrityError("transaction receipt operation crosses its identity or commit group boundary")
        object_payload = operation_payload.get("context_object")
        if (
            not isinstance(object_payload, dict)
            or str(object_payload.get("tenant_id") or "default") != tenant_value
            or str(object_payload.get("owner_user_id") or "") != user_value
        ):
            raise ReceiptIntegrityError("transaction receipt object crosses tenant or owner boundary")
    expected_snapshots = effect_snapshots(stored_operations, relation_effects=relation_effects)
    snapshot_uris = [str(item.get("uri") or "") for item in expected_snapshots]
    if not all(snapshot_uris) or len(snapshot_uris) != len(set(snapshot_uris)):
        raise ReceiptIntegrityError("transaction receipt contains duplicate object effects")
    if canonical_json(snapshots) != canonical_json(expected_snapshots):
        raise ReceiptIntegrityError("transaction receipt effect snapshots are corrupt")
    if payload.get("planned_effect_digests") != [item["effect_digest"] for item in expected_snapshots]:
        raise ReceiptIntegrityError("transaction receipt effect digest set is corrupt")
    if payload.get("before_revisions") != {str(item["uri"]): item["before_revision"] for item in expected_snapshots}:
        raise ReceiptIntegrityError("transaction receipt before revisions are corrupt")
    if payload.get("after_revisions") != {str(item["uri"]): item["after_revision"] for item in expected_snapshots}:
        raise ReceiptIntegrityError("transaction receipt after revisions are corrupt")
    if not isinstance(payload.get("planning_digest"), str) or not payload["planning_digest"]:
        raise ReceiptIntegrityError("transaction receipt planning digest is missing")
    if not isinstance(payload.get("prepared_intent_digest"), str) or not payload["prepared_intent_digest"]:
        raise ReceiptIntegrityError("transaction receipt prepared-intent digest is missing")
    prepared_intent_schema_version = payload.get("prepared_intent_schema_version")
    if prepared_intent_schema_version is not None and (
        not isinstance(prepared_intent_schema_version, str) or not prepared_intent_schema_version
    ):
        raise ReceiptIntegrityError("transaction receipt prepared-intent schema identity is invalid")
    for effect in relation_effects:
        if not isinstance(effect, dict) or not isinstance(effect.get("identity"), dict):
            raise ReceiptIntegrityError("transaction receipt relation effect is invalid")
        identity = {
            "source_uri": str(effect["identity"].get("source_uri") or ""),
            "relation_type": str(effect["identity"].get("relation_type") or ""),
            "target_uri": str(effect["identity"].get("target_uri") or ""),
        }
        expected_exists = effect.get("expected_exists")
        if not all(identity.values()) or not isinstance(expected_exists, bool):
            raise ReceiptIntegrityError("transaction receipt relation identity is incomplete")
        if expected_exists:
            relation = effect.get("relation")
            if not isinstance(relation, dict):
                raise ReceiptIntegrityError("transaction receipt relation snapshot is missing")
            normalized = {
                **identity,
                "weight": float(relation.get("weight", 1.0)),
                "metadata": dict(relation.get("metadata", {}) or {}),
            }
            if effect.get("relation_digest") != canonical_digest(normalized):
                raise ReceiptIntegrityError("transaction receipt relation snapshot is corrupt")
        elif effect.get("relation_digest") != canonical_digest(identity):
            raise ReceiptIntegrityError("transaction receipt relation absence proof is corrupt")
    return payload


def receipt_snapshot(payload: dict[str, Any], uri: str) -> dict[str, Any]:
    matches = [
        item for item in payload.get("effect_snapshots", []) if isinstance(item, dict) and item.get("uri") == uri
    ]
    if len(matches) != 1:
        raise ReceiptIntegrityError("transaction receipt does not contain exactly one requested effect")
    return dict(matches[0])
