"""Durable idempotency markers for ordinary Context operations."""

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
    EFFECT_MARKER_SCHEMA_VERSION,
    EffectProofError,
    build_marker,
    normalized_relation,
    object_effect_from_store,
    relation_effects_from_manifest,
    relation_identity,
    validate_marker,
)
from memoryos.operations.commit.redo_log import RedoIntegrityError
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus


class OperationMarkerStore:
    @staticmethod
    def _operation_marker(committer, operation_id: str) -> Path:
        key = require_safe_path_segment(operation_id, "operation_id")
        return committer.artifact_root / "system" / "operations" / f"{key}.json"

    @staticmethod
    def _regular_lock_keys(operation: ContextOperation) -> tuple[str, ...]:
        target = operation.target_uri or f"{operation.user_id}:{operation.operation_id}"
        replacement = OperationMarkerStore._replacement_uri(operation)
        return tuple(sorted({target, f"operation-id:{operation.operation_id}", *([replacement] if replacement else [])}))

    @staticmethod
    def _replacement_uri(operation: ContextOperation) -> str:
        if operation.action != OperationAction.SUPERSEDE:
            return ""
        payload = operation.payload.get("context_object")
        return str(payload.get("uri") or "") if isinstance(payload, dict) else ""

    @staticmethod
    def _write_operation_marker(
        committer,
        operation: ContextOperation,
        *,
        source_effect: dict | None,
        relation_manifest: dict | None,
        diff: ContextDiff,
    ) -> None:
        committer._validate_regular_recovery_effect(
            operation.user_id,
            operation,
            source_effect,
            relation_manifest=relation_manifest,
        )
        path = committer._operation_marker(operation.operation_id)
        committer._reject_control_symlink(path, "operation marker")
        if path.exists():
            committer._validate_operation_marker(path, operation)
            return
        stored = operation.to_dict()
        stored["status"] = OperationStatus.COMMITTED.value
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
            relation_effects=(
                []
                if operation.action == OperationAction.DELETE
                else relation_effects_from_manifest(relation_manifest)
            ),
            diff=diff.to_dict(),
            operations=[stored],
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
                "operation": stored,
            }
        )
        core = {key: value for key, value in payload.items() if key != "marker_digest"}
        payload["marker_digest"] = canonical_digest(core)
        committer._atomic_create_json(path, payload, artifact_root=committer.artifact_root)

    @staticmethod
    def _validate_operation_marker(committer, path: Path, operation: ContextOperation) -> ContextOperation:
        committer._validate_and_bind_operations(operation.user_id, [operation])
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
        raw = payload.get("operation")
        if not isinstance(raw, dict):
            raise ValueError("operation marker is missing its persisted operation")
        stored = ContextOperation.from_dict(raw)
        committer._validate_and_bind_operations(operation.user_id, [stored])
        requested = operation
        if requested.target_uri is None and stored.target_uri is not None:
            requested = ContextOperation.from_dict(operation.to_dict())
            requested.target_uri = stored.target_uri
        fingerprint = payload.get("effect_fingerprint")
        if (
            operation.target_uri not in {None, stored.target_uri}
            or payload.get("target_uri") != stored.target_uri
            or fingerprint != committer._operation_effect_fingerprint(stored)
            or fingerprint != committer._operation_effect_fingerprint(requested)
        ):
            raise ValueError("operation idempotency marker conflicts with the requested effect")
        stored.status = OperationStatus.COMMITTED
        return stored

    @staticmethod
    def _refresh_regular_effect_proofs(committer, changed_uris: list[str]) -> None:
        wanted = set(changed_uris)
        root = committer.artifact_root / "system" / "operations"
        if not wanted or not root.exists():
            return
        for path in sorted(root.glob("*.json")):
            try:
                if path.is_symlink():
                    raise ValueError("operation marker cannot be a symbolic link")
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                quarantine_control_file(
                    committer.artifact_root,
                    path,
                    kind="operation_marker",
                    error=exc,
                    identifiers={"operation_id": path.stem},
                )
                raise RedoIntegrityError("operation marker is unreadable") from exc
            core = {key: value for key, value in payload.items() if key != "marker_digest"}
            if payload.get("schema_version") != EFFECT_MARKER_SCHEMA_VERSION or payload.get("marker_digest") != canonical_digest(core):
                quarantine_control_file(
                    committer.artifact_root,
                    path,
                    kind="operation_marker",
                    error=ValueError("marker integrity mismatch"),
                    identifiers={"operation_id": path.stem},
                )
                raise RedoIntegrityError("operation marker integrity check failed")
            effects = payload.get("object_effects")
            if not isinstance(effects, list) or not any(
                isinstance(effect, dict) and str(effect.get("uri") or "") in wanted for effect in effects
            ):
                continue
            refreshed = []
            for effect in effects:
                if not isinstance(effect, dict) or str(effect.get("uri") or "") not in wanted:
                    refreshed.append(effect)
                    continue
                uri = str(effect["uri"])
                logical_absence = str(effect.get("absence_mode") or "") == "logical_deleted"
                refreshed.append(
                    object_effect_from_store(
                        committer.source_store,
                        uri,
                        operation_type=str(effect.get("operation_type") or "update"),
                        expected_exists=not logical_absence,
                        logical_absence=logical_absence,
                    )
                )
            payload["object_effects"] = refreshed
            relation_effects = payload.get("relation_effects")
            if committer.relation_store is not None and isinstance(relation_effects, list):
                updated_relations = []
                for effect in relation_effects:
                    if not isinstance(effect, dict) or effect.get("expected_exists") is not True:
                        updated_relations.append(effect)
                        continue
                    identity = relation_identity(dict(effect.get("identity", {}) or {}))
                    if not ({identity["source_uri"], identity["target_uri"]} & wanted):
                        updated_relations.append(effect)
                        continue
                    matches = [
                        relation
                        for relation in committer.relation_store.relations_of(
                            identity["source_uri"], tenant_id=committer.tenant_id
                        )
                        if relation_identity(relation.to_dict()) == identity
                    ]
                    if len(matches) != 1:
                        updated_relations.append(effect)
                        continue
                    normalized = normalized_relation(matches[0])
                    updated_relations.append(
                        {
                            **effect,
                            "identity": identity,
                            **identity,
                            "relation": normalized,
                            "relation_digest": canonical_digest(normalized),
                        }
                    )
                payload["relation_effects"] = updated_relations
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
                if key not in {"target_resolution_reason", "target_candidates", "projection_tombstone_ids"}
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


__all__ = ["OperationMarkerStore"]
