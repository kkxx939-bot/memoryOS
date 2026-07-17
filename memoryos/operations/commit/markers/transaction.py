"""Implementation component for TransactionMarkerStore.

The public OperationCommitter delegates explicitly to this component so fault
injection hooks remain available on the facade.
"""

from __future__ import annotations

import json
from pathlib import Path

from memoryos.core.durable_io.quarantine import quarantine_control_file
from memoryos.core.ids import require_safe_path_segment, stable_hash
from memoryos.core.integrity import canonical_json
from memoryos.operations.commit.effect_marker import (
    EffectProofError,
    relation_effects_from_manifest,
    validate_marker,
)
from memoryos.operations.commit.outbox_envelope import (
    OutboxIntegrityError,
    validate_outbox,
)
from memoryos.operations.commit.planning_proof import (
    CANONICAL_PREPARED_INTENT_SCHEMA_VERSION,
    PlanningProofIntegrityError,
)
from memoryos.operations.commit.receipt import (
    TRANSACTION_RECEIPT_SCHEMA_VERSION,
    ReceiptIntegrityError,
    build_transaction_receipt,
    validate_transaction_receipt,
)
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation


class TransactionMarkerStore:
    """Own the TransactionMarkerStore responsibility of a commit."""

    @staticmethod
    def _transaction_marker(committer, idempotency_key: str) -> Path:
        key = require_safe_path_segment(idempotency_key, "canonical idempotency_key")
        return committer.artifact_root / "system" / "transactions" / f"{key}.json"

    @staticmethod
    def _outbox_path(committer, transaction_id: str) -> Path:
        key = require_safe_path_segment(transaction_id, "canonical transaction_id")
        return committer.artifact_root / "system" / "outbox" / f"{key}.json"

    @staticmethod
    def _reject_control_symlink(path: Path, label: str) -> None:
        if path.is_symlink():
            raise ValueError(f"{label} cannot be a symbolic link")

    @staticmethod
    def _write_transaction_marker(
        committer,
        path: Path,
        diff: ContextDiff,
        operations: list[ContextOperation],
        *,
        relation_manifests: dict[str, dict] | None = None,
    ) -> None:
        if not operations:
            raise ValueError("canonical transaction marker requires operations")
        keys = {committer._validate_canonical_artifact_keys(operation)[1] for operation in operations}
        if len(keys) != 1 or path != committer._transaction_marker(next(iter(keys))):
            raise ValueError("canonical transaction marker path does not match its operations")
        committer._reject_control_symlink(path, "canonical transaction receipt")
        if path.exists():
            committer._validate_transaction_marker(path, operations)
            return
        transaction_ids = {committer._validate_canonical_artifact_keys(operation)[0] for operation in operations}
        if len(transaction_ids) != 1:
            raise ValueError("canonical transaction marker requires one transaction id")
        relation_effects = committer._marker_relation_effects(relation_manifests)
        outbox_path = committer._outbox_path(next(iter(transaction_ids)))
        committer._reject_control_symlink(outbox_path, "canonical outbox")
        if not outbox_path.exists():
            raise ValueError("canonical receipt requires its previously published prepared outbox intent")
        try:
            outbox = validate_outbox(
                json.loads(outbox_path.read_text(encoding="utf-8")),
                transaction_id=next(iter(transaction_ids)),
                idempotency_key=next(iter(keys)),
                tenant_id=committer.tenant_id,
                user_id=operations[0].user_id,
                operations=operations,
                allowed_statuses={"prepared", "source_committed"},
            )
            immutable_intent = committer.planning_proofs.load_canonical_intent(
                next(iter(transaction_ids)),
                operations=operations,
                prepared_intent_digest=str(outbox["prepared_intent_digest"]),
            )
            intent_digest = str(immutable_intent["prepared_intent_digest"])
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            OutboxIntegrityError,
            PlanningProofIntegrityError,
        ) as exc:
            raise ValueError("canonical receipt requires a valid prepared intent") from exc
        planning_digests = {str(operation.payload.get("planning_digest") or "") for operation in operations}
        if len(planning_digests) != 1 or "" in planning_digests:
            raise ValueError("canonical receipt requires exactly one planning digest")
        payload = build_transaction_receipt(
            transaction_id=next(iter(transaction_ids)),
            idempotency_key=next(iter(keys)),
            tenant_id=committer.tenant_id,
            user_id=operations[0].user_id,
            commit_group_id=next(
                (
                    str(operation.payload.get("commit_group_id") or "")
                    for operation in operations
                    if operation.payload.get("commit_group_id")
                ),
                "",
            ),
            operations=operations,
            diff=diff.to_dict(),
            planning_digest=next(iter(planning_digests)),
            prepared_intent_digest=intent_digest,
            prepared_intent_schema_version=CANONICAL_PREPARED_INTENT_SCHEMA_VERSION,
            relation_effects=relation_effects,
            created_at=diff.created_at,
        )
        committer._reject_control_symlink(path, "canonical transaction receipt")
        committer._atomic_create_json(path, payload, artifact_root=committer.artifact_root)

    @staticmethod
    def _validate_transaction_marker(
        committer,
        path: Path,
        operations: list[ContextOperation],
    ) -> ContextDiff:
        if not operations:
            raise ValueError("canonical transaction marker validation requires operations")
        keys = {committer._validate_canonical_artifact_keys(operation)[1] for operation in operations}
        if len(keys) != 1 or path != committer._transaction_marker(next(iter(keys))):
            raise ValueError("canonical transaction marker path does not match its operations")
        transaction_ids = {committer._validate_canonical_artifact_keys(operation)[0] for operation in operations}
        if len(transaction_ids) != 1:
            raise ValueError("canonical transaction marker requires one transaction id")
        try:
            payload = validate_marker(
                path,
                committer.source_store,
                committer.relation_store,
                transaction_id=next(iter(transaction_ids)),
                idempotency_key=next(iter(keys)),
                tenant_id=committer.tenant_id,
                user_id=operations[0].user_id,
                operation_ids=[operation.operation_id for operation in operations],
            )
        except EffectProofError as exc:
            if path.exists():
                quarantine_control_file(
                    committer.artifact_root,
                    path,
                    kind="transaction_marker",
                    error=exc,
                    identifiers={
                        "transaction_id": next(iter(transaction_ids)),
                        "idempotency_key": next(iter(keys)),
                    },
                )
            raise ValueError("canonical transaction marker cannot prove its durable effect") from exc
        diff_payload = payload.get("diff")
        if not isinstance(diff_payload, dict):
            raise ValueError("canonical transaction marker is missing its persisted diff")
        diff = committer._diff_from_payload(diff_payload)
        committer._validate_and_bind_operations(operations[0].user_id, operations)
        committer._validate_and_bind_operations(operations[0].user_id, diff.operations)
        if diff.user_id != operations[0].user_id:
            raise ValueError("canonical transaction marker crosses a user boundary")
        if committer._canonical_transaction_request_fingerprint(
            diff.operations
        ) != committer._canonical_transaction_request_fingerprint(
            operations
        ) or committer._canonical_transaction_effect_fingerprint(
            diff.operations
        ) != committer._canonical_transaction_effect_fingerprint(operations):
            raise ValueError("canonical idempotency marker conflicts with the requested transaction")
        return diff

    @staticmethod
    def _validate_transaction_marker_tenant(committer, path: Path) -> None:
        if path.is_symlink():
            raise ValueError("canonical transaction receipt cannot be a symbolic link")
        payload = json.loads(path.read_text(encoding="utf-8"))
        tenant = committer._validate_tenant_id(payload["tenant_id"], "canonical transaction marker tenant_id")
        if tenant != committer.tenant_id:
            raise ValueError("canonical transaction marker crosses the bound tenant")

    @staticmethod
    def _transaction_marker_diff(committer, path: Path) -> ContextDiff:
        if path.is_symlink():
            raise ValueError("canonical transaction receipt cannot be a symbolic link")
        payload = json.loads(path.read_text(encoding="utf-8"))
        diff_payload = payload.get("diff")
        if not isinstance(diff_payload, dict):
            raise ValueError("canonical transaction marker is missing its persisted diff")
        operations_payload = payload.get("operations")
        if payload.get("schema_version") not in {
            "effect_marker_v1",
            TRANSACTION_RECEIPT_SCHEMA_VERSION,
        } or not isinstance(operations_payload, list):
            raise ValueError("canonical transaction marker schema is unsupported")
        if payload.get("schema_version") == TRANSACTION_RECEIPT_SCHEMA_VERSION:
            try:
                validate_transaction_receipt(payload)
            except ReceiptIntegrityError as exc:
                raise ValueError("canonical transaction receipt is corrupt") from exc
        return committer._diff_from_payload(diff_payload)

    @staticmethod
    def _marker_relation_effects(
        committer,
        relation_manifests: dict[str, dict] | None,
    ) -> list[dict]:
        if not relation_manifests:
            return []
        by_identity: dict[str, dict] = {}
        for operation_id in sorted(relation_manifests):
            for effect in relation_effects_from_manifest(relation_manifests[operation_id]):
                identity_key = canonical_json(effect["identity"])
                current = by_identity.get(identity_key)
                if current is None or effect["expected_exists"] is True:
                    by_identity[identity_key] = effect
        return [by_identity[key] for key in sorted(by_identity)]

    @staticmethod
    def _canonical_transaction_request_fingerprint(committer, operations: list[ContextOperation]) -> str:
        normalized = []
        for operation in sorted(operations, key=lambda item: item.operation_id):
            payload = json.loads(json.dumps(operation.to_dict(), ensure_ascii=False))
            payload.pop("status", None)
            payload.pop("created_at", None)
            committer._strip_relation_timestamps(payload)
            normalized.append(payload)
        canonical_json(normalized)
        return stable_hash(normalized, length=64)

    @staticmethod
    def _canonical_transaction_request_fingerprint_v2(committer, operations: list[ContextOperation]) -> str:
        normalized = []
        for operation in sorted(operations, key=lambda item: item.operation_id):
            payload = operation.to_dict()
            payload.pop("status", None)
            normalized.append(payload)
        canonical_json(normalized)
        return stable_hash(normalized, length=64)

    @staticmethod
    def _canonical_transaction_effect_fingerprint(committer, operations: list[ContextOperation]) -> str:
        effects = []
        for operation in sorted(operations, key=lambda item: item.operation_id):
            effects.append(
                {
                    "operation_id": operation.operation_id,
                    "user_id": operation.user_id,
                    "context_type": operation.context_type.value,
                    "action": operation.action.value,
                    "target_uri": operation.target_uri,
                    "context_object": committer._context_object_without_relation_timestamps(
                        operation.payload.get("context_object")
                    ),
                    "content": operation.payload.get("content", ""),
                }
            )
        canonical_json(effects)
        return stable_hash(effects, length=64)

    @staticmethod
    def _canonical_transaction_effect_fingerprint_v2(committer, operations: list[ContextOperation]) -> str:
        effects = []
        for operation in sorted(operations, key=lambda item: item.operation_id):
            effects.append(
                {
                    "operation_id": operation.operation_id,
                    "user_id": operation.user_id,
                    "context_type": operation.context_type.value,
                    "action": operation.action.value,
                    "target_uri": operation.target_uri,
                    "context_object": operation.payload.get("context_object"),
                    "content": operation.payload.get("content", ""),
                }
            )
        canonical_json(effects)
        return stable_hash(effects, length=64)

    @staticmethod
    def _strip_relation_timestamps(committer, operation_payload: dict) -> None:
        payload = operation_payload.get("payload")
        if not isinstance(payload, dict):
            return
        context_object = payload.get("context_object")
        payload["context_object"] = committer._context_object_without_relation_timestamps(context_object)

    @staticmethod
    def _context_object_without_relation_timestamps(committer, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = json.loads(json.dumps(value, ensure_ascii=False))
        relations = normalized.get("relations")
        if isinstance(relations, list):
            for relation in relations:
                if isinstance(relation, dict):
                    relation.pop("created_at", None)
        return normalized
