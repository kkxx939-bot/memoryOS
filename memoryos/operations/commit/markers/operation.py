"""Implementation component for OperationMarkerStore.

The public OperationCommitter delegates explicitly to this component so fault
injection hooks remain available on the facade.
"""

from __future__ import annotations

import json
from pathlib import Path

from memoryos.contextdb.model.context_layer import ContextLayers
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.core.durable_io import atomic_write_json
from memoryos.core.durable_io.quarantine import quarantine_control_file
from memoryos.core.ids import require_safe_path_segment, stable_hash
from memoryos.core.integrity import canonical_digest, canonical_json
from memoryos.operations.commit.effect_marker import (
    EffectProofError,
    build_marker,
    normalized_relation,
    object_effect_from_store,
    relation_effects_from_manifest,
    relation_identity,
    validate_marker,
)
from memoryos.operations.commit.planning_proof import (
    PENDING_PREPARED_INTENT_SCHEMA_VERSION,
    PlanningProofIntegrityError,
)
from memoryos.operations.commit.receipt import (
    TRANSACTION_RECEIPT_SCHEMA_VERSION,
    ReceiptIntegrityError,
    build_transaction_receipt,
    load_transaction_receipt,
    validate_transaction_receipt,
)
from memoryos.operations.commit.redo_log import RedoIntegrityError
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus


class OperationMarkerStore:
    """Own the OperationMarkerStore responsibility of a commit."""

    @staticmethod
    def _operation_marker(committer, operation_id: str) -> Path:
        key = require_safe_path_segment(operation_id, "operation_id")
        return committer.artifact_root / "system" / "operations" / f"{key}.json"

    @staticmethod
    def _regular_lock_keys(operation: ContextOperation) -> tuple[str, ...]:
        """Fence both mutable target state and the immutable operation identity."""

        target = operation.target_uri or f"{operation.user_id}:{operation.operation_id}"
        return tuple(sorted({target, f"operation-id:{operation.operation_id}"}))

    @staticmethod
    def _write_operation_marker(
        committer,
        operation: ContextOperation,
        *,
        source_effect: dict | None,
        relation_manifest: dict | None,
        diff: ContextDiff,
    ) -> None:
        if operation.payload.get("canonical_memory") is True:
            return
        committer._validate_regular_recovery_effect(
            operation.user_id,
            operation,
            source_effect,
            relation_manifest=relation_manifest,
        )
        path = committer._operation_marker(operation.operation_id)
        committer._reject_control_symlink(path, "operation receipt")
        if operation.payload.get("canonical_pending_proposal") is True:
            committer._bind_pending_receipt_identity(operation)
            if path.exists():
                stored = committer._validate_operation_marker(path, operation)
                committer._publish_pending_current_head(path, stored)
                committer._mark_current_heads_published([operation])
                return
            planning_digest = committer._ensure_pending_planning_digest(operation)
            try:
                intent = committer.planning_proofs.load_pending_intent(
                    operation.operation_id,
                    operation=operation,
                    relation_manifest=relation_manifest,
                )
            except PlanningProofIntegrityError as exc:
                raise ValueError("pending receipt requires its pre-write prepared intent") from exc
            intent_digest = str(intent["prepared_intent_digest"])
            receipt = build_transaction_receipt(
                transaction_id=operation.operation_id,
                idempotency_key=str(operation.payload.get("idempotency_key") or operation.operation_id),
                tenant_id=committer.tenant_id,
                user_id=operation.user_id,
                commit_group_id=str(operation.payload.get("commit_group_id") or ""),
                operations=[operation],
                diff=diff.to_dict(),
                planning_digest=planning_digest,
                prepared_intent_digest=intent_digest,
                prepared_intent_schema_version=PENDING_PREPARED_INTENT_SCHEMA_VERSION,
                relation_effects=relation_effects_from_manifest(relation_manifest),
                created_at=diff.created_at,
            )
            committer._notify("before_receipt", operation.operation_id)
            committer._reject_control_symlink(path, "pending operation receipt")
            committer._atomic_create_json(path, receipt, artifact_root=committer.artifact_root)
            committer._notify("after_receipt", operation.operation_id)
            committer._notify("before_current_head", operation.operation_id)
            committer._publish_canonical_current_heads(path, receipt)
            committer._mark_current_heads_published([operation])
            committer._notify("after_current_head", operation.operation_id)
            return
        stored_operation = operation.to_dict()
        stored_operation["status"] = OperationStatus.COMMITTED.value
        if path.exists():
            committer._validate_operation_marker(path, operation)
            return
        object_effects = []
        for uri in committer._regular_source_effect_uris(operation):
            logical_absence = operation.action == OperationAction.DELETE and uri == operation.target_uri
            object_effects.append(
                object_effect_from_store(
                    committer.source_store,
                    uri,
                    operation_type=operation.action.value,
                    expected_exists=not logical_absence,
                    logical_absence=logical_absence,
                )
            )
        payload = build_marker(
            transaction_id=operation.operation_id,
            idempotency_key=operation.operation_id,
            tenant_id=committer.tenant_id,
            user_id=operation.user_id,
            operation_ids=[operation.operation_id],
            object_effects=object_effects,
            # A regular DELETE retires derived RelationStore rows through its
            # receipt-bound projection tombstones after the Source effect is
            # committed.  Binding their transient pre-cleanup presence into
            # the immutable Source marker would make a successful delete
            # impossible to replay once that outbox has done its job.
            relation_effects=(
                [] if operation.action == OperationAction.DELETE else relation_effects_from_manifest(relation_manifest)
            ),
            diff=diff.to_dict(),
            operations=[stored_operation],
        )
        payload.update(
            {
                "operation_id": operation.operation_id,
                "action": operation.action.value,
                "context_type": operation.context_type.value,
                "target_uri": operation.target_uri,
                "commit_group_id": operation.payload.get("commit_group_id"),
                "commit_consumer": operation.payload.get("commit_consumer"),
                "effect_fingerprint": committer._operation_effect_fingerprint(operation),
                "operation": stored_operation,
            }
        )
        core = {key: value for key, value in payload.items() if key != "marker_digest"}
        payload["marker_digest"] = canonical_digest(core)
        committer._reject_control_symlink(path, "operation marker")
        committer._atomic_create_json(path, payload, artifact_root=committer.artifact_root)

    @staticmethod
    def _bind_pending_receipt_identity(committer, operation: ContextOperation) -> None:
        """Bind a pending lifecycle operation before Source/diff/receipt publication."""

        commit_group_id = operation.payload.get("commit_group_id")
        if not isinstance(commit_group_id, str) or not commit_group_id:
            raise ValueError("pending lifecycle operation requires a commit group identity")
        operation.payload.update(
            {
                "transaction_id": operation.operation_id,
                "idempotency_key": str(operation.payload.get("idempotency_key") or operation.operation_id),
                "tenant_id": committer.tenant_id,
            }
        )

    @staticmethod
    def _publish_pending_current_head(
        committer,
        path: Path,
        operation: ContextOperation,
    ) -> None:
        if operation.payload.get("canonical_pending_proposal") is not True:
            return
        try:
            receipt = load_transaction_receipt(path)
        except ReceiptIntegrityError as exc:
            raise ValueError("pending operation receipt is corrupt") from exc
        committer._publish_canonical_current_heads(path, receipt)

    @staticmethod
    def _validate_operation_marker(committer, path: Path, operation: ContextOperation) -> ContextOperation:
        committer._validate_and_bind_operations(operation.user_id, [operation])
        committer._reject_control_symlink(path, "operation receipt")
        try:
            raw_payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("operation marker is unreadable") from exc
        if raw_payload.get("schema_version") == TRANSACTION_RECEIPT_SCHEMA_VERSION:
            try:
                receipt = validate_transaction_receipt(
                    raw_payload,
                    transaction_id=operation.operation_id,
                    tenant_id=committer.tenant_id,
                    user_id=operation.user_id,
                    operation_ids=[operation.operation_id],
                )
            except ReceiptIntegrityError as exc:
                raise ValueError("pending operation receipt is corrupt") from exc
            stored_payloads = receipt.get("operations", [])
            if len(stored_payloads) != 1 or not isinstance(stored_payloads[0], dict):
                raise ValueError("pending operation receipt has invalid membership")
            stored = ContextOperation.from_dict(stored_payloads[0])
            committer._validate_and_bind_operations(operation.user_id, [stored])
            requested = operation
            if requested.target_uri is None and stored.target_uri is not None:
                requested = ContextOperation.from_dict(operation.to_dict())
                requested.target_uri = stored.target_uri
            if committer._operation_effect_fingerprint(stored) != committer._operation_effect_fingerprint(requested):
                raise ValueError("operation idempotency receipt conflicts with the requested effect")
            if stored.payload.get("canonical_pending_proposal") is True:
                committer._ensure_pending_planning_digest(stored)
                try:
                    committer.planning_proofs.load_pending_intent(
                        stored.operation_id,
                        operation=stored,
                        prepared_intent_digest=str(receipt.get("prepared_intent_digest") or ""),
                    )
                except PlanningProofIntegrityError as exc:
                    raise ValueError("pending operation receipt is detached from its prepared intent") from exc
            stored.status = OperationStatus.COMMITTED
            return stored
        try:
            payload = validate_marker(
                path,
                committer.source_store,
                committer.relation_store,
                transaction_id=operation.operation_id,
                idempotency_key=operation.operation_id,
                tenant_id=committer.tenant_id,
                user_id=operation.user_id,
                operation_ids=[operation.operation_id],
            )
        except EffectProofError as exc:
            if path.exists():
                quarantine_control_file(
                    committer.artifact_root,
                    path,
                    kind="operation_marker",
                    error=exc,
                    identifiers={"operation_id": operation.operation_id},
                )
            raise ValueError("operation marker cannot prove its durable effect") from exc
        expected = {
            "operation_id": operation.operation_id,
            "action": operation.action.value,
            "context_type": operation.context_type.value,
            "commit_group_id": operation.payload.get("commit_group_id"),
            "commit_consumer": operation.payload.get("commit_consumer"),
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("operation idempotency marker conflicts with the requested operation")
        stored_payload = payload.get("operation")
        if not isinstance(stored_payload, dict):
            raise ValueError("operation idempotency marker is missing its persisted operation")
        stored = ContextOperation.from_dict(stored_payload)
        committer._validate_and_bind_operations(operation.user_id, [stored])
        if operation.target_uri not in {None, stored.target_uri} or payload.get("target_uri") != stored.target_uri:
            raise ValueError("operation idempotency marker conflicts with the requested target")
        requested = operation
        if requested.target_uri is None and stored.target_uri is not None:
            requested = ContextOperation.from_dict(operation.to_dict())
            requested.target_uri = stored.target_uri
        if payload.get("effect_fingerprint") != committer._operation_effect_fingerprint(stored) or payload.get(
            "effect_fingerprint"
        ) != committer._operation_effect_fingerprint(requested):
            raise ValueError("operation idempotency marker conflicts with the requested effect")
        stored.status = OperationStatus.COMMITTED
        return stored

    @staticmethod
    def _refresh_regular_effect_proofs(committer, changed_uris: list[str]) -> None:
        """Atomically advance prior regular markers to the current Source fact."""

        wanted = set(changed_uris)
        marker_root = committer.artifact_root / "system" / "operations"
        if not wanted or not marker_root.exists():
            return
        for path in sorted(marker_root.glob("*.json")):
            try:
                if path.is_symlink():
                    raise OSError("operation marker cannot be a symbolic link")
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                if path.exists():
                    quarantine_control_file(
                        committer.artifact_root,
                        path,
                        kind="operation_marker",
                        error=exc,
                        identifiers={"operation_id": path.stem},
                    )
                raise RedoIntegrityError("regular operation marker is unreadable") from exc
            if isinstance(payload, dict) and payload.get("schema_version") == TRANSACTION_RECEIPT_SCHEMA_VERSION:
                # Immutable pending receipts are historical facts; a later
                # lifecycle revision must never refresh their effect proof.
                continue
            if not isinstance(payload, dict) or payload.get("schema_version") != "effect_marker_v1":
                quarantine_control_file(
                    committer.artifact_root,
                    path,
                    kind="operation_marker",
                    error=ValueError("unsupported marker schema"),
                    identifiers={"operation_id": path.stem},
                )
                raise RedoIntegrityError("regular operation marker schema is unsupported")
            digest = payload.get("marker_digest")
            core = {key: value for key, value in payload.items() if key != "marker_digest"}
            if not isinstance(digest, str) or digest != canonical_digest(core):
                quarantine_control_file(
                    committer.artifact_root,
                    path,
                    kind="operation_marker",
                    error=ValueError("marker digest mismatch"),
                    identifiers={"operation_id": path.stem},
                )
                raise RedoIntegrityError("regular operation marker digest is corrupt")
            effects = payload.get("object_effects")
            if not isinstance(effects, list) or not any(
                isinstance(effect, dict) and str(effect.get("uri") or "") in wanted for effect in effects
            ):
                continue
            refreshed: list[dict] = []
            for effect in effects:
                if (
                    not isinstance(effect, dict)
                    or str(effect.get("uri") or "") not in wanted
                    or effect.get("expected_exists") is not True
                ):
                    refreshed.append(effect)
                    continue
                refreshed.append(
                    object_effect_from_store(
                        committer.source_store,
                        str(effect["uri"]),
                        operation_type=str(effect.get("operation_type") or "UPDATE"),
                    )
                )
            payload["object_effects"] = refreshed
            relation_effects = payload.get("relation_effects")
            if committer.relation_store is not None and isinstance(relation_effects, list):
                refreshed_relations: list[dict] = []
                for effect in relation_effects:
                    if not isinstance(effect, dict) or effect.get("expected_exists") is not True:
                        refreshed_relations.append(effect)
                        continue
                    identity = relation_identity(dict(effect.get("identity", {}) or {}))
                    if not ({identity["source_uri"], identity["target_uri"]} & wanted):
                        refreshed_relations.append(effect)
                        continue
                    matches = [
                        relation
                        for relation in committer.relation_store.relations_of(
                            identity["source_uri"],
                            tenant_id=committer.tenant_id,
                        )
                        if relation.source_uri == identity["source_uri"]
                        and relation.relation_type == identity["relation_type"]
                        and relation.target_uri == identity["target_uri"]
                    ]
                    if len(matches) != 1:
                        refreshed_relations.append(effect)
                        continue
                    normalized = normalized_relation(matches[0])
                    refreshed_relations.append(
                        {
                            **effect,
                            "identity": identity,
                            **identity,
                            "relation": normalized,
                            "relation_digest": canonical_digest(normalized),
                        }
                    )
                payload["relation_effects"] = refreshed_relations
            updated_core = {key: value for key, value in payload.items() if key != "marker_digest"}
            payload["marker_digest"] = canonical_digest(updated_core)
            atomic_write_json(path, payload, artifact_root=committer.artifact_root)

    @staticmethod
    def _operation_effect_fingerprint(committer, operation: ContextOperation) -> str:
        if operation.action in {OperationAction.ADD, OperationAction.UPDATE, OperationAction.MERGE}:
            effect_payload = {
                "context_object": committer._normalized_regular_object_effect(operation),
                "content": operation.payload.get("content", ""),
            }
        elif operation.action == OperationAction.SUPERSEDE:
            effect_payload = {
                "context_object": committer._normalized_regular_object_effect(operation),
                "content": operation.payload.get("content", ""),
                "reason": operation.payload.get("reason", operation.payload.get("supersede_reason", "")),
            }
        else:
            effect_payload = {
                key: value
                for key, value in operation.payload.items()
                if key
                not in {
                    "target_resolution_reason",
                    "target_candidates",
                    # Internal durable outbox bindings are receipt-covered but
                    # are not part of the caller's semantic delete request.
                    # Excluding them lets a retry with the same operation id
                    # load the exact persisted binding from its marker.
                    "projection_tombstone_ids",
                }
            }
        effect = {
            "operation_id": operation.operation_id,
            "user_id": operation.user_id,
            "context_type": operation.context_type.value,
            "action": operation.action.value,
            "target_uri": operation.target_uri,
            "effect_payload": effect_payload,
        }
        canonical_json(effect)
        return stable_hash(effect, length=64)

    @staticmethod
    def _normalized_regular_object_effect(committer, operation: ContextOperation) -> object:
        payload = operation.payload.get("context_object")
        if not isinstance(payload, dict):
            return payload
        obj = committer._materialize_action_policy_source_relations(ContextObject.from_dict(payload))
        if operation.payload.get("content"):
            obj.layers = ContextLayers(
                l0_uri=f"{obj.uri}/.abstract.md",
                l1_uri=f"{obj.uri}/.overview.md",
                l2_uri=f"{obj.uri}/content.md",
            )
        return obj.to_dict()
