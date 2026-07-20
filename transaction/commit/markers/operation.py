"""普通 Context 操作的耐久幂等标记。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from foundation.ids import require_safe_path_segment, stable_hash
from foundation.integrity import canonical_digest, canonical_json
from infrastructure.store.model.context.context_object import ContextObject
from transaction.commit.control import RedoIntegrityError
from transaction.commit.effect_proof import (
    EFFECT_MARKER_SCHEMA_VERSION,
    EffectProofError,
    build_marker,
    normalized_relation,
    object_effect_from_store,
    relation_effects_from_manifest,
    relation_identity,
    validate_marker,
)
from transaction.model.context_diff import ContextDiff
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_action import OperationAction
from transaction.model.operation_status import OperationStatus

if TYPE_CHECKING:
    from transaction.commit.host import OperationTransactionHost


class OperationMarkerStore:
    def _operation_marker(self: OperationTransactionHost, operation_id: str) -> Path:
        require_safe_path_segment(operation_id, "operation_id")
        return self.marker_store.path(operation_id)

    def _regular_lock_keys(self: OperationTransactionHost, operation: ContextOperation) -> tuple[str, ...]:
        target = operation.target_uri or f"{operation.user_id}:{operation.operation_id}"
        replacement = self._replacement_uri(operation)
        return tuple(
            sorted({target, f"operation-id:{operation.operation_id}", *([replacement] if replacement else [])})
        )

    def _replacement_uri(self: OperationTransactionHost, operation: ContextOperation) -> str:
        if operation.action != OperationAction.SUPERSEDE:
            return ""
        payload = operation.payload.get("context_object")
        return str(payload.get("uri") or "") if isinstance(payload, dict) else ""

    def _write_operation_marker(
        self: OperationTransactionHost,
        operation: ContextOperation,
        *,
        source_effect: dict | None,
        relation_manifest: dict | None,
        diff: ContextDiff,
    ) -> None:
        self._validate_regular_recovery_effect(
            operation.user_id,
            operation,
            source_effect,
            relation_manifest=relation_manifest,
        )
        path = self._operation_marker(operation.operation_id)
        self._reject_control_symlink(path, "operation marker")
        if path.exists():
            self._validate_operation_marker(path, operation)
            return
        object_effects = []
        for uri in self._regular_source_effect_uris(operation):
            logical_absence = operation.action == OperationAction.DELETE and uri == operation.target_uri
            object_effects.append(
                object_effect_from_store(
                    self.source_store,
                    uri,
                    operation_type=operation.action.value,
                    expected_exists=not logical_absence,
                    logical_absence=logical_absence,
                )
            )
        payload = build_marker(
            transaction_id=operation.operation_id,
            idempotency_key=operation.operation_id,
            tenant_id=self.tenant_id,
            user_id=operation.user_id,
            operation_ids=[operation.operation_id],
            object_effects=object_effects,
            relation_effects=(
                [] if operation.action == OperationAction.DELETE else relation_effects_from_manifest(relation_manifest)
            ),
            diff_id=diff.diff_id,
            operation_fingerprints={
                operation.operation_id: self._operation_effect_fingerprint(operation),
            },
        )
        payload.update(
            {
                "operation_id": operation.operation_id,
                "action": operation.action.value,
                "context_type": operation.context_type.value,
                "target_uri": operation.target_uri,
                "commit_group_id": operation.payload.get("commit_group_id"),
                "commit_consumer": operation.payload.get("commit_consumer"),
                "effect_fingerprint": self._operation_effect_fingerprint(operation),
            }
        )
        core = {key: value for key, value in payload.items() if key != "marker_digest"}
        payload["marker_digest"] = canonical_digest(core)
        self.marker_store.create(operation.operation_id, payload)

    def _validate_operation_marker(
        self: OperationTransactionHost, path: Path, operation: ContextOperation
    ) -> ContextOperation:
        self._validate_and_bind_operations(operation.user_id, [operation])
        try:
            payload = validate_marker(
                path,
                self.source_store,
                self.relation_store,
                transaction_id=operation.operation_id,
                idempotency_key=operation.operation_id,
                tenant_id=self.tenant_id,
                user_id=operation.user_id,
                operation_ids=[operation.operation_id],
            )
        except EffectProofError as exc:
            if path.exists():
                self.marker_store.quarantine(
                    path,
                    exc,
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
        requested = ContextOperation.from_dict(operation.to_dict())
        stored_target = payload.get("target_uri")
        if requested.target_uri is None and isinstance(stored_target, str) and stored_target:
            requested = ContextOperation.from_dict(operation.to_dict())
            requested.target_uri = stored_target
        self._validate_and_bind_operations(operation.user_id, [requested])
        fingerprint = payload.get("effect_fingerprint")
        fingerprints = payload.get("operation_fingerprints")
        if (
            not isinstance(fingerprints, dict)
            or fingerprints.get(operation.operation_id) != fingerprint
            or operation.target_uri not in {None, requested.target_uri}
            or payload.get("target_uri") != requested.target_uri
            or fingerprint != self._operation_effect_fingerprint(requested)
        ):
            raise ValueError("operation idempotency marker conflicts with the requested effect")
        requested.status = OperationStatus.COMMITTED
        return requested

    def _refresh_regular_effect_proofs(self: OperationTransactionHost, changed_uris: list[str]) -> None:
        wanted = set(changed_uris)
        if not wanted:
            return
        for path in self.marker_store.paths():
            try:
                payload = self.marker_store.read(path)
            except (OSError, UnicodeError, ValueError) as exc:
                self.marker_store.quarantine(
                    path,
                    exc,
                    identifiers={"operation_id": path.stem},
                )
                raise RedoIntegrityError("operation marker is unreadable") from exc
            core = {key: value for key, value in payload.items() if key != "marker_digest"}
            if payload.get("schema_version") != EFFECT_MARKER_SCHEMA_VERSION or payload.get(
                "marker_digest"
            ) != canonical_digest(core):
                self.marker_store.quarantine(
                    path,
                    ValueError("marker integrity mismatch"),
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
                        self.source_store,
                        uri,
                        operation_type=str(effect.get("operation_type") or "update"),
                        expected_exists=not logical_absence,
                        logical_absence=logical_absence,
                    )
                )
            payload["object_effects"] = refreshed
            relation_effects = payload.get("relation_effects")
            if self.relation_store is not None and isinstance(relation_effects, list):
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
                        for relation in self.relation_store.relations_of(
                            identity["source_uri"], tenant_id=self.tenant_id
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
            self.marker_store.replace(path, payload)

    def _operation_effect_fingerprint(self: OperationTransactionHost, operation: ContextOperation) -> str:
        if operation.action in {OperationAction.ADD, OperationAction.UPDATE, OperationAction.MERGE}:
            effect_payload = {
                "context_object": self._normalized_regular_object_effect(operation),
                "content": operation.payload.get("content", ""),
            }
        elif operation.action == OperationAction.SUPERSEDE:
            effect_payload = {
                "context_object": self._normalized_regular_object_effect(operation),
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

    def _normalized_regular_object_effect(self: OperationTransactionHost, operation: ContextOperation) -> object:
        payload = operation.payload.get("context_object")
        if not isinstance(payload, dict):
            return payload
        obj = self._materialize_domain_object(ContextObject.from_dict(payload))
        if self.context_effects is not None:
            obj = self.context_effects.prepare_object(
                obj,
                str(operation.payload.get("content", "")),
            )
        return obj.to_dict()


__all__ = ["OperationMarkerStore"]
