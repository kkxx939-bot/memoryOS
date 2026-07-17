"""Implementation component for CanonicalEffectExecutor.

The public OperationCommitter delegates explicitly to this component so fault
injection hooks remain available on the facade.
"""

from __future__ import annotations

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.core.errors import RevisionConflictError
from memoryos.core.ids import stable_hash
from memoryos.core.integrity import canonical_digest, canonical_json
from memoryos.operations.commit.outbox_envelope import (
    planned_effect_manifest,
)
from memoryos.operations.commit.redo_log import RedoIntegrityError
from memoryos.operations.model.context_operation import ContextOperation


class CanonicalEffectExecutor:
    """Own the CanonicalEffectExecutor responsibility of a commit."""

    @staticmethod
    def _build_canonical_relation_manifest(
        committer,
        operation: ContextOperation,
        before_object: ContextObject | None,
    ) -> dict:
        payload = operation.payload.get("context_object")
        if not isinstance(payload, dict):
            raise ValueError("canonical relation manifest requires context_object")
        desired = ContextObject.from_dict(payload)
        expected = committer._canonical_relation_specs(operation, desired)
        expected_keys = {committer._relation_spec_key(spec) for spec in expected}
        previous_keys = committer._canonical_managed_relation_keys(before_object) if committer.relation_store is not None else []
        remove = committer._unique_relation_keys(
            [key for key in previous_keys if committer._relation_spec_key(key) not in expected_keys]
        )
        core = {
            "schema_version": "canonical_relation_manifest_v1",
            "operation_id": operation.operation_id,
            "user_id": operation.user_id,
            "tenant_id": str(operation.payload.get("tenant_id") or "default"),
            "transaction_id": str(operation.payload.get("transaction_id") or ""),
            "idempotency_key": str(operation.payload.get("idempotency_key") or ""),
            "target_uri": operation.target_uri,
            "expected": expected,
            "remove": remove,
        }
        return {**core, "fingerprint": stable_hash(core, length=64)}

    @staticmethod
    def _validate_canonical_relation_manifest(
        committer,
        operation: ContextOperation,
        manifest: dict,
    ) -> None:
        if manifest.get("schema_version") != "canonical_relation_manifest_v1":
            raise RedoIntegrityError("canonical relation manifest schema is unsupported")
        core = {key: value for key, value in manifest.items() if key != "fingerprint"}
        if manifest.get("fingerprint") != stable_hash(core, length=64):
            raise RedoIntegrityError("canonical relation manifest fingerprint is corrupt")
        if (
            manifest.get("operation_id") != operation.operation_id
            or manifest.get("user_id") != operation.user_id
            or manifest.get("tenant_id") != str(operation.payload.get("tenant_id") or "default")
            or manifest.get("transaction_id") != str(operation.payload.get("transaction_id") or "")
            or manifest.get("idempotency_key") != str(operation.payload.get("idempotency_key") or "")
            or manifest.get("target_uri") != operation.target_uri
            or not isinstance(manifest.get("expected"), list)
            or not isinstance(manifest.get("remove"), list)
        ):
            raise RedoIntegrityError("canonical relation manifest crosses its operation boundary")

    @staticmethod
    def _apply_canonical_relation_manifest(
        committer,
        operation: ContextOperation,
        manifest: dict,
    ) -> None:
        committer._validate_canonical_relation_manifest(operation, manifest)
        if committer.relation_store is None:
            if manifest.get("expected") or manifest.get("remove"):
                raise RedoIntegrityError("canonical relation manifest requires a RelationStore")
            return
        for key in manifest.get("remove", []) or []:
            committer.relation_store.delete_relation(
                str(key["source_uri"]),
                str(key["relation_type"]),
                str(key["target_uri"]),
                tenant_id=str(manifest["tenant_id"]),
            )
        committer._ensure_relation_specs([dict(item) for item in manifest.get("expected", []) or []])
        committer._validate_canonical_relation_manifest_effect(manifest)

    @staticmethod
    def _validate_canonical_relation_manifest_effect(committer, manifest: dict) -> None:
        if committer.relation_store is None:
            if manifest.get("expected") or manifest.get("remove"):
                raise RedoIntegrityError("canonical relation effect has no RelationStore")
            return
        for spec in manifest.get("expected", []) or []:
            actual = {
                canonical_json(committer._relation_effect_spec(relation))
                for relation in committer.relation_store.relations_of(
                    str(spec["source_uri"]),
                    tenant_id=str(manifest["tenant_id"]),
                )
            }
            if canonical_json(spec) not in actual:
                raise RedoIntegrityError("canonical RelationStore effect is incomplete")
        for key in manifest.get("remove", []) or []:
            if any(
                relation.source_uri == key["source_uri"]
                and relation.relation_type == key["relation_type"]
                and relation.target_uri == key["target_uri"]
                for relation in committer.relation_store.relations_of(
                    str(key["source_uri"]),
                    tenant_id=str(manifest["tenant_id"]),
                )
            ):
                raise RedoIntegrityError("canonical RelationStore retained a removed managed relation")

    @staticmethod
    def _canonical_relation_specs(
        committer,
        operation: ContextOperation,
        obj: ContextObject,
    ) -> list[dict]:
        if committer.relation_store is None:
            return []
        metadata = dict(obj.metadata or {})
        relation_metadata = {
            "tenant_id": obj.tenant_id or "default",
            "owner_user_id": obj.owner_user_id,
            "canonical_transaction_id": operation.payload.get("transaction_id"),
            "canonical_idempotency_key": operation.payload.get("idempotency_key"),
            "source_revision": metadata.get("revision"),
            "commit_group_id": operation.payload.get("commit_group_id"),
        }
        specs = []
        for relation in obj.relations:
            specs.append(
                committer._relation_spec(
                    relation.source_uri,
                    relation.relation_type,
                    relation.target_uri,
                    {**dict(relation.metadata or {}), **relation_metadata},
                    weight=relation.weight,
                )
            )
        kind = str(metadata.get("canonical_kind") or "")
        if kind == "claim":
            slot_uri = obj.uri.rsplit("/claims/", 1)[0]
            specs.append(committer._relation_spec(obj.uri, "belongs_to_slot", slot_uri, relation_metadata))
        elif kind == "slot":
            specs.extend(
                committer._relation_spec(
                    obj.uri,
                    "has_claim",
                    f"{obj.uri}/claims/{claim_id}",
                    relation_metadata,
                )
                for claim_id in sorted(str(item) for item in metadata.get("claim_ids", []) or [] if str(item))
            )
        return committer._unique_relation_specs(specs)

    @staticmethod
    def _canonical_managed_relation_keys(
        committer,
        obj: ContextObject | None,
    ) -> list[dict]:
        if obj is None:
            return []
        keys = [committer._relation_key_payload(committer._relation_effect_spec(relation)) for relation in obj.relations]
        metadata = dict(obj.metadata or {})
        kind = str(metadata.get("canonical_kind") or "")
        if kind == "claim":
            slot_uri = obj.uri.rsplit("/claims/", 1)[0]
            keys.append(committer._relation_key_payload(committer._relation_spec(obj.uri, "belongs_to_slot", slot_uri, {})))
        elif kind == "slot":
            keys.extend(
                committer._relation_key_payload(
                    committer._relation_spec(
                        obj.uri,
                        "has_claim",
                        f"{obj.uri}/claims/{claim_id}",
                        {},
                    )
                )
                for claim_id in sorted(str(item) for item in metadata.get("claim_ids", []) or [] if str(item))
            )
        return committer._unique_relation_keys(keys)

    @staticmethod
    def _validate_existing_canonical_effect(committer, operation: ContextOperation) -> None:
        payload = operation.payload.get("context_object")
        if not isinstance(payload, dict):
            raise ValueError("canonical operation requires context_object")
        desired = ContextObject.from_dict(payload)
        current = committer.source_store.read_object(desired.uri)
        if canonical_json(current.to_dict()) != canonical_json(desired.to_dict()):
            raise RevisionConflictError(
                f"canonical recovery found a divergent object at desired revision: {desired.uri}"
            )
        expected_content = str(operation.payload.get("content", ""))
        try:
            actual_content = committer.source_store.read_content(current.layers.l2_uri or current.uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            actual_content = ""
        if actual_content != expected_content:
            raise RevisionConflictError(
                f"canonical recovery found divergent content at desired revision: {desired.uri}"
            )

    @staticmethod
    def _capture_canonical_source_effect(
        committer,
        operation: ContextOperation,
        relation_manifest: dict,
    ) -> dict:
        committer._validate_canonical_relation_manifest(operation, relation_manifest)
        committer._validate_existing_canonical_effect(operation)
        committer._validate_canonical_relation_manifest_effect(relation_manifest)
        planned = planned_effect_manifest(operation, relation_manifest)
        core = {
            "schema_version": "canonical_source_effect_v1",
            "operation_id": operation.operation_id,
            "transaction_id": str(operation.payload.get("transaction_id") or ""),
            "idempotency_key": str(operation.payload.get("idempotency_key") or ""),
            "tenant_id": committer.tenant_id,
            "user_id": operation.user_id,
            "uri": planned["uri"],
            "object_digest": planned["object_digest"],
            "content_digest": planned["content_digest"],
            "revision": planned["revision"],
            "relation_manifest_digest": planned["relation_manifest_digest"],
            "planned_effect_digest": planned["effect_digest"],
        }
        return {**core, "effect_digest": canonical_digest(core)}

    @staticmethod
    def _validate_canonical_source_effect(
        committer,
        operation: ContextOperation,
        source_effect: dict | None,
        relation_manifest: dict | None,
    ) -> None:
        if not isinstance(source_effect, dict) or not isinstance(relation_manifest, dict):
            raise RedoIntegrityError("canonical redo is missing its Source or Relation effect")
        stored_core = {key: value for key, value in source_effect.items() if key != "effect_digest"}
        if source_effect.get("schema_version") != "canonical_source_effect_v1" or source_effect.get(
            "effect_digest"
        ) != canonical_digest(stored_core):
            raise RedoIntegrityError("canonical redo Source effect digest is corrupt")
        try:
            actual = committer._capture_canonical_source_effect(operation, relation_manifest)
        except (FileNotFoundError, RevisionConflictError, ValueError) as exc:
            raise RedoIntegrityError("canonical redo Source effect does not match durable state") from exc
        if actual != source_effect:
            raise RedoIntegrityError("canonical redo Source effect does not match durable state")
