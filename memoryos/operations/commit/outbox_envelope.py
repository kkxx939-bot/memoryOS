"""Strict canonical transaction outbox envelopes and state transitions."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Any

from memoryos.core.ids import require_safe_path_segment
from memoryos.core.integrity import canonical_digest, canonical_json
from memoryos.operations.model.context_operation import ContextOperation

OUTBOX_SCHEMA_VERSION = "canonical_outbox_v1"
OUTBOX_EVENT_TYPE = "MemoryTransaction"
OUTBOX_STATES = {"prepared", "source_committed", "committed", "aborted"}
OUTBOX_TERMINAL_STATES = {"committed", "aborted"}
OUTBOX_TRANSITIONS = {
    "prepared": {"prepared", "source_committed", "aborted"},
    "source_committed": {"source_committed", "committed", "aborted"},
    "committed": {"committed"},
    "aborted": {"aborted"},
}


def projection_workspace_id(
    operations: Sequence[ContextOperation | dict[str, Any]],
) -> str:
    """Return the single workspace that owns a canonical projection job.

    Multiple or absent workspace refs intentionally collapse to the empty
    workspace so queue health conservatively blocks every workspace-scoped
    CURRENT read for that owner until replay completes.
    """

    workspace_ids: set[str] = set()
    for operation in operations:
        operation_payload = operation.payload if isinstance(operation, ContextOperation) else operation.get("payload")
        if not isinstance(operation_payload, dict):
            continue
        raw_object = operation_payload.get("context_object")
        if not isinstance(raw_object, dict):
            continue
        metadata = raw_object.get("metadata")
        if not isinstance(metadata, dict):
            continue
        scope = metadata.get("scope")
        if not isinstance(scope, dict):
            scope = {}
        for value in (
            raw_object.get("workspace_id"),
            metadata.get("workspace_id"),
            metadata.get("project_id"),
            scope.get("workspace_id"),
            scope.get("project_id"),
        ):
            if isinstance(value, str) and value.strip():
                workspace_ids.add(value.strip())
        applicability = scope.get("applicability")
        if not isinstance(applicability, dict):
            continue
        refs = applicability.get("all_of")
        if not isinstance(refs, list | tuple):
            continue
        for ref in refs:
            if not isinstance(ref, dict) or str(ref.get("kind") or "") not in {
                "workspace",
                "project",
            }:
                continue
            value = ref.get("id")
            if isinstance(value, str) and value.strip():
                workspace_ids.add(value.strip())
    return next(iter(workspace_ids)) if len(workspace_ids) == 1 else ""


PREPARED_INTENT_FIELDS = (
    "schema_version",
    "event_type",
    "transaction_id",
    "idempotency_key",
    "tenant_id",
    "user_id",
    "operation_ids",
    "operations",
    "operations_digest",
    "before_images",
    "before_images_digest",
    "effect_manifests",
    "effect_manifests_digest",
    "claim_revisions",
    "commit_group_id",
)


class OutboxIntegrityError(RuntimeError):
    """An outbox envelope is corrupt or crosses its transaction boundary."""


def normalized_operation(operation: ContextOperation | dict[str, Any]) -> dict[str, Any]:
    payload = operation.to_dict() if isinstance(operation, ContextOperation) else dict(operation)
    payload.pop("status", None)
    return payload


def operation_set_digest(operations: Sequence[ContextOperation | dict[str, Any]]) -> str:
    normalized = sorted(
        (normalized_operation(operation) for operation in operations),
        key=lambda item: str(item.get("operation_id") or ""),
    )
    return canonical_digest(normalized)


def planned_effect_manifest(
    operation: ContextOperation,
    relation_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    object_payload = operation.payload.get("context_object")
    if not isinstance(object_payload, dict) or not object_payload.get("uri"):
        raise OutboxIntegrityError("canonical operation is missing its planned object effect")
    metadata = dict(object_payload.get("metadata", {}) or {})
    relation_payload = dict(relation_manifest or {})
    core = {
        "operation_id": operation.operation_id,
        "transaction_id": str(operation.payload.get("transaction_id") or ""),
        "idempotency_key": str(operation.payload.get("idempotency_key") or ""),
        "tenant_id": str(operation.payload.get("tenant_id") or "default"),
        "user_id": operation.user_id,
        "operation_type": operation.action.value,
        "uri": str(object_payload["uri"]),
        "expected_exists": True,
        "object_digest": canonical_digest(object_payload),
        "content_digest": canonical_digest(str(operation.payload.get("content", ""))),
        "revision": metadata.get("revision"),
        "relation_manifest": relation_payload,
        "relation_manifest_digest": canonical_digest(relation_payload),
    }
    return {**core, "effect_digest": canonical_digest(core)}


def build_outbox(
    *,
    transaction_id: str,
    idempotency_key: str,
    tenant_id: str,
    user_id: str,
    operations: list[ContextOperation],
    status: str,
    before_images: list[dict[str, Any]],
    effect_manifests: list[dict[str, Any]],
    claim_revisions: list[dict[str, Any]],
    commit_group_id: str,
    receipt_path: str = "",
    receipt_digest: str = "",
) -> dict[str, Any]:
    if status not in OUTBOX_STATES:
        raise OutboxIntegrityError("canonical outbox status is unsupported")
    operation_commit_groups = {str(operation.payload.get("commit_group_id") or "") for operation in operations}
    if not isinstance(commit_group_id, str) or not commit_group_id or operation_commit_groups != {commit_group_id}:
        raise OutboxIntegrityError("canonical outbox crosses or omits its commit group boundary")
    normalized_operations = [normalized_operation(operation) for operation in operations]
    immutable_intent: dict[str, Any] = {
        "schema_version": OUTBOX_SCHEMA_VERSION,
        "event_type": OUTBOX_EVENT_TYPE,
        "transaction_id": transaction_id,
        "idempotency_key": idempotency_key,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "operation_ids": [operation.operation_id for operation in operations],
        "operations": normalized_operations,
        "operations_digest": operation_set_digest(normalized_operations),
        "before_images": before_images,
        "before_images_digest": canonical_digest(before_images),
        "effect_manifests": effect_manifests,
        "effect_manifests_digest": canonical_digest(effect_manifests),
        "claim_revisions": claim_revisions,
        "commit_group_id": commit_group_id,
    }
    core: dict[str, Any] = {
        **immutable_intent,
        "status": status,
        "prepared_intent_digest": canonical_digest(immutable_intent),
        "receipt_path": receipt_path,
        "receipt_digest": receipt_digest,
    }
    return {**core, "outbox_digest": canonical_digest(core)}


def validate_outbox(
    payload: object,
    *,
    transaction_id: str | None = None,
    idempotency_key: str | None = None,
    tenant_id: str | None = None,
    user_id: str | None = None,
    operations: list[ContextOperation] | None = None,
    allowed_statuses: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise OutboxIntegrityError("canonical outbox must be a JSON object")
    if payload.get("schema_version") != OUTBOX_SCHEMA_VERSION:
        raise OutboxIntegrityError("canonical outbox schema is unsupported")
    if payload.get("event_type") != OUTBOX_EVENT_TYPE:
        raise OutboxIntegrityError("canonical outbox event type is invalid")
    status = payload.get("status")
    if status not in OUTBOX_STATES or (allowed_statuses is not None and status not in allowed_statuses):
        raise OutboxIntegrityError("canonical outbox status is invalid for this operation")
    digest = payload.get("outbox_digest")
    core = {key: value for key, value in payload.items() if key != "outbox_digest"}
    if not isinstance(digest, str) or digest != canonical_digest(core):
        raise OutboxIntegrityError("canonical outbox digest is corrupt")
    for key, expected in (
        ("transaction_id", transaction_id),
        ("idempotency_key", idempotency_key),
        ("tenant_id", tenant_id),
        ("user_id", user_id),
    ):
        if expected is not None and payload.get(key) != expected:
            raise OutboxIntegrityError(f"canonical outbox {key} does not match")
    operation_ids = payload.get("operation_ids")
    stored_operations = payload.get("operations")
    before_images = payload.get("before_images")
    effect_manifests = payload.get("effect_manifests")
    claim_revisions = payload.get("claim_revisions")
    if (
        not isinstance(operation_ids, list)
        or not operation_ids
        or len(set(str(item) for item in operation_ids)) != len(operation_ids)
        or not isinstance(stored_operations, list)
        or len(stored_operations) != len(operation_ids)
        or not isinstance(before_images, list)
        or not isinstance(effect_manifests, list)
        or len(effect_manifests) != len(operation_ids)
        or not isinstance(claim_revisions, list)
    ):
        raise OutboxIntegrityError("canonical outbox transaction members are incomplete")
    stored_ids = [str(item.get("operation_id") or "") for item in stored_operations if isinstance(item, dict)]
    effect_ids = [str(item.get("operation_id") or "") for item in effect_manifests if isinstance(item, dict)]
    if stored_ids != [str(item) for item in operation_ids] or set(effect_ids) != set(stored_ids):
        raise OutboxIntegrityError("canonical outbox operation ids are inconsistent")
    stored_commit_groups = {
        str(dict(item.get("payload", {}) or {}).get("commit_group_id") or "")
        for item in stored_operations
        if isinstance(item, dict)
    }
    if (
        not isinstance(payload.get("commit_group_id"), str)
        or not payload.get("commit_group_id")
        or stored_commit_groups != {str(payload["commit_group_id"])}
    ):
        raise OutboxIntegrityError("canonical outbox crosses or omits its commit group boundary")
    if payload.get("operations_digest") != operation_set_digest(stored_operations):
        raise OutboxIntegrityError("canonical outbox operations digest is corrupt")
    if payload.get("before_images_digest") != canonical_digest(before_images):
        raise OutboxIntegrityError("canonical outbox before images are corrupt")
    if payload.get("effect_manifests_digest") != canonical_digest(effect_manifests):
        raise OutboxIntegrityError("canonical outbox effect manifests are corrupt")
    expected_claim_revisions: list[dict[str, Any]] = []
    for operation in stored_operations:
        assert isinstance(operation, dict)
        operation_payload = operation.get("payload")
        context_object = operation_payload.get("context_object") if isinstance(operation_payload, dict) else None
        if not isinstance(context_object, dict):
            continue
        metadata = dict(context_object.get("metadata", {}) or {})
        if metadata.get("canonical_kind") != "claim":
            continue
        uri = context_object.get("uri")
        claim_id = metadata.get("claim_id")
        revision = metadata.get("revision")
        if (
            not isinstance(uri, str)
            or not uri
            or not isinstance(claim_id, str)
            or not claim_id
            or isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 1
        ):
            raise OutboxIntegrityError("canonical outbox Claim operation has an invalid revision identity")
        expected_claim_revisions.append({"uri": uri, "claim_id": claim_id, "revision": revision})
    if claim_revisions != expected_claim_revisions:
        raise OutboxIntegrityError("canonical outbox claim revision set is detached from its immutable operations")
    immutable_intent = {key: payload.get(key) for key in PREPARED_INTENT_FIELDS}
    if payload.get("prepared_intent_digest") != canonical_digest(immutable_intent):
        raise OutboxIntegrityError("canonical outbox prepared intent digest is corrupt")
    receipt_path = payload.get("receipt_path", "")
    receipt_digest = payload.get("receipt_digest", "")
    if status == "committed":
        if (
            not isinstance(receipt_path, str)
            or not receipt_path
            or not isinstance(receipt_digest, str)
            or not receipt_digest
        ):
            raise OutboxIntegrityError("committed canonical outbox requires an immutable receipt reference")
        try:
            receipt_identity = require_safe_path_segment(
                payload.get("idempotency_key"),
                "canonical outbox idempotency_key",
            )
        except (TypeError, ValueError) as exc:
            raise OutboxIntegrityError("canonical outbox receipt identity is invalid") from exc
        normalized_receipt = PurePosixPath(receipt_path)
        expected_receipt = PurePosixPath("system") / "transactions" / f"{receipt_identity}.json"
        if normalized_receipt.is_absolute() or normalized_receipt != expected_receipt:
            raise OutboxIntegrityError("committed canonical outbox must reference its unique immutable receipt")
    elif receipt_path or receipt_digest:
        raise OutboxIntegrityError("pre-commit canonical outbox cannot reference a later receipt")
    for effect in effect_manifests:
        _validate_effect_manifest(effect, payload)
    stored_by_id = {
        str(item["operation_id"]): ContextOperation.from_dict(item)
        for item in stored_operations
        if isinstance(item, dict) and item.get("operation_id")
    }
    for effect in effect_manifests:
        assert isinstance(effect, dict)
        operation = stored_by_id.get(str(effect.get("operation_id") or ""))
        if operation is None or effect != planned_effect_manifest(
            operation,
            dict(effect.get("relation_manifest", {}) or {}),
        ):
            raise OutboxIntegrityError("canonical outbox effect does not match its operation")
    effect_uris = {str(effect["uri"]) for effect in effect_manifests if isinstance(effect, dict)}
    before_uris: set[str] = set()
    for snapshot in before_images:
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("exists"), bool):
            raise OutboxIntegrityError("canonical outbox before image is invalid")
        uri = str(snapshot.get("uri") or "")
        if not uri or uri in before_uris:
            raise OutboxIntegrityError("canonical outbox before image identity is invalid")
        before_uris.add(uri)
        object_payload = snapshot.get("object")
        relations = snapshot.get("relations")
        if not isinstance(relations, list) or snapshot.get("relations_digest") != canonical_digest(relations):
            raise OutboxIntegrityError("canonical outbox before-image relations are corrupt")
        if snapshot["exists"]:
            if (
                not isinstance(object_payload, dict)
                or str(object_payload.get("uri") or "") != uri
                or str(object_payload.get("tenant_id") or "default") != payload.get("tenant_id")
                or str(object_payload.get("owner_user_id") or "") not in {"", payload.get("user_id")}
                or not isinstance(snapshot.get("content"), str)
            ):
                raise OutboxIntegrityError("canonical outbox before image crosses its boundary")
        elif object_payload is not None:
            raise OutboxIntegrityError("canonical outbox absent before image contains an object")
    if before_uris != effect_uris and (before_uris or payload.get("status") != "committed"):
        raise OutboxIntegrityError("canonical outbox before images are incomplete")
    if operations is not None:
        requested_ids = [operation.operation_id for operation in operations]
        if set(requested_ids) != {str(item) for item in operation_ids} or len(requested_ids) != len(operation_ids):
            raise OutboxIntegrityError("canonical outbox is missing requested operation ids")
        if operation_set_digest(operations) != payload.get("operations_digest"):
            raise OutboxIntegrityError("canonical outbox operations do not match redo")
    return payload


def assert_transition(previous: str, requested: str) -> None:
    if requested not in OUTBOX_TRANSITIONS.get(previous, set()):
        raise OutboxIntegrityError(f"canonical outbox terminal state cannot transition from {previous} to {requested}")


def load_outbox_text(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise OutboxIntegrityError("canonical outbox JSON is corrupt") from exc
    return validate_outbox(payload)


def _validate_effect_manifest(effect: object, envelope: dict[str, Any]) -> None:
    if not isinstance(effect, dict):
        raise OutboxIntegrityError("canonical outbox effect manifest is invalid")
    core = {key: value for key, value in effect.items() if key != "effect_digest"}
    if effect.get("effect_digest") != canonical_digest(core):
        raise OutboxIntegrityError("canonical outbox effect manifest digest is corrupt")
    if (
        effect.get("transaction_id") != envelope.get("transaction_id")
        or effect.get("idempotency_key") != envelope.get("idempotency_key")
        or effect.get("tenant_id") != envelope.get("tenant_id")
        or effect.get("user_id") != envelope.get("user_id")
        or effect.get("operation_id") not in envelope.get("operation_ids", [])
        or not effect.get("uri")
        or effect.get("expected_exists") is not True
        or effect.get("relation_manifest_digest") != canonical_digest(effect.get("relation_manifest", {}))
    ):
        raise OutboxIntegrityError("canonical outbox effect manifest crosses its transaction boundary")


def same_immutable_envelope(left: dict[str, Any], right: dict[str, Any]) -> bool:
    mutable = {"status", "outbox_digest", "receipt_path", "receipt_digest"}
    return canonical_json({k: v for k, v in left.items() if k not in mutable}) == canonical_json(
        {k: v for k, v in right.items() if k not in mutable}
    )


def prepared_intent_digest(payload: dict[str, Any]) -> str:
    """Return the immutable prepared-intent identity used by receipts."""

    validated = validate_outbox(payload)
    return str(validated["prepared_intent_digest"])


def prepared_intent_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the complete immutable intent carried by an outbox envelope."""

    validated = validate_outbox(payload)
    return {key: validated[key] for key in PREPARED_INTENT_FIELDS}
