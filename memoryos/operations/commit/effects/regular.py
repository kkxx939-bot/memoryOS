"""Implementation component for RegularEffectExecutor.

The public OperationCommitter delegates explicitly to this component so fault
injection hooks remain available on the facade.
"""

from __future__ import annotations

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.ordinary_relations import (
    OrdinaryRelationEligibility,
    ordinary_relation_serving_eligibility,
    ordinary_relation_specs_for_object,
)
from memoryos.core.ids import stable_hash
from memoryos.core.integrity import canonical_json, text_digest
from memoryos.operations.commit.redo_log import RedoIntegrityError
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class RegularEffectExecutor:
    """Own the RegularEffectExecutor responsibility of a commit."""

    @staticmethod
    def _capture_regular_source_effect(
        committer,
        operation: ContextOperation,
        relation_manifest: dict | None = None,
    ) -> dict:
        uris = committer._regular_source_effect_uris(operation)
        snapshots = []
        for uri in uris:
            try:
                obj = committer.source_store.read_object(uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                snapshots.append({"uri": uri, "exists": False})
                continue
            layer_hashes: dict[str, str | None] = {}
            layer_uris = tuple(
                dict.fromkeys(
                    item for item in (obj.layers.l0_uri, obj.layers.l1_uri, obj.layers.l2_uri or obj.uri) if item
                )
            )
            for layer_uri in layer_uris:
                try:
                    content = committer.source_store.read_content(str(layer_uri))
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                    layer_hashes[str(layer_uri)] = None
                else:
                    layer_hashes[str(layer_uri)] = text_digest(content)
            snapshots.append(
                {
                    "uri": uri,
                    "exists": True,
                    "object": obj.to_dict(),
                    "layer_hashes": layer_hashes,
                }
            )
        core = {
            "schema_version": "regular_source_effect_v2",
            "operation_id": operation.operation_id,
            "user_id": operation.user_id,
            "uris": uris,
            "snapshots": snapshots,
            "relations": (
                list(relation_manifest.get("expected", []) or [])
                if isinstance(relation_manifest, dict)
                else committer._expected_regular_relation_specs(operation)
            ),
            "relation_manifest_fingerprint": (
                str(relation_manifest.get("fingerprint") or "") if isinstance(relation_manifest, dict) else ""
            ),
        }
        return {**core, "fingerprint": stable_hash(core, length=64)}

    @staticmethod
    def _validate_regular_recovery_effect(
        committer,
        user_id: str,
        operation: ContextOperation,
        source_effect: dict | None,
        *,
        require_relation_presence: bool = True,
        relation_manifest: dict | None = None,
    ) -> None:
        if not isinstance(source_effect, dict):
            raise RedoIntegrityError("regular redo entry is missing its SourceStore effect")
        if source_effect.get("schema_version") != "regular_source_effect_v2":
            raise RedoIntegrityError("regular redo SourceStore effect schema is unsupported")
        stored_core = {key: value for key, value in source_effect.items() if key != "fingerprint"}
        if source_effect.get("fingerprint") != stable_hash(stored_core, length=64):
            raise RedoIntegrityError("regular redo SourceStore effect fingerprint is corrupt")
        if (
            source_effect.get("operation_id") != operation.operation_id
            or source_effect.get("user_id") != user_id
            or list(source_effect.get("uris", []) or []) != committer._regular_source_effect_uris(operation)
        ):
            raise RedoIntegrityError("regular redo SourceStore effect is bound to another operation")
        actual = committer._capture_regular_source_effect(operation, relation_manifest)
        if actual.get("fingerprint") != source_effect.get("fingerprint"):
            raise RedoIntegrityError("regular redo SourceStore effect does not match durable state")
        expected_tenant = committer._regular_operation_tenant(operation)
        committer._validate_regular_action_postcondition(operation, actual)
        if relation_manifest is not None:
            committer._validate_regular_relation_manifest(operation, relation_manifest)
        elif committer.relation_store is not None:
            raise RedoIntegrityError("regular redo entry is missing its relation manifest")
        expected_relations = (
            list(relation_manifest.get("expected", []) or [])
            if isinstance(relation_manifest, dict)
            else committer._expected_regular_relation_specs(operation)
        )
        if source_effect.get("relations") != expected_relations:
            raise RedoIntegrityError("regular redo relation effect does not match its operation")
        if source_effect.get("relation_manifest_fingerprint", "") != (
            str(relation_manifest.get("fingerprint") or "") if isinstance(relation_manifest, dict) else ""
        ):
            raise RedoIntegrityError("regular redo relation manifest does not match its SourceStore effect")
        for snapshot in actual.get("snapshots", []) or []:
            if not snapshot.get("exists") or not isinstance(snapshot.get("object"), dict):
                raise RedoIntegrityError("regular redo SourceStore effect is missing its target object")
            obj = ContextObject.from_dict(snapshot["object"])
            try:
                parsed = ContextURI.parse(obj.uri)
            except (TypeError, ValueError) as exc:
                raise RedoIntegrityError("regular redo SourceStore URI is invalid") from exc
            if parsed.authority == "user":
                if parsed.user_id != user_id or obj.owner_user_id != user_id:
                    raise RedoIntegrityError("regular redo SourceStore effect crosses a user boundary")
            elif obj.owner_user_id not in {None, "", user_id}:
                raise RedoIntegrityError("regular redo SourceStore effect crosses an owner boundary")
            if str(obj.tenant_id or "default") != expected_tenant:
                raise RedoIntegrityError("regular redo SourceStore effect crosses a tenant boundary")
            if obj.context_type != operation.context_type:
                raise RedoIntegrityError("regular redo SourceStore effect changes context type")
        if require_relation_presence:
            if isinstance(relation_manifest, dict):
                committer._validate_regular_relation_manifest_effect(relation_manifest)
            else:
                committer._validate_regular_relation_postcondition(expected_relations)

    @staticmethod
    def _validate_and_restore_regular_recovery_effect(
        committer,
        user_id: str,
        operation: ContextOperation,
        source_effect: dict | None,
        relation_manifest: dict | None,
    ) -> None:
        committer._validate_regular_recovery_effect(
            user_id,
            operation,
            source_effect,
            require_relation_presence=False,
            relation_manifest=relation_manifest,
        )
        if isinstance(relation_manifest, dict):
            committer._apply_regular_relation_manifest(operation, relation_manifest)
        else:
            assert isinstance(source_effect, dict)
            committer._restore_regular_relation_effect(operation, source_effect)
        committer._validate_regular_recovery_effect(
            user_id,
            operation,
            source_effect,
            relation_manifest=relation_manifest,
        )

    @staticmethod
    def _validate_regular_action_postcondition(
        committer,
        operation: ContextOperation,
        effect: dict,
    ) -> None:
        snapshots = {
            str(snapshot.get("uri") or ""): snapshot
            for snapshot in effect.get("snapshots", []) or []
            if isinstance(snapshot, dict)
        }

        def required(uri: str) -> tuple[ContextObject, dict]:
            snapshot = snapshots.get(uri)
            if snapshot is None or not snapshot.get("exists") or not isinstance(snapshot.get("object"), dict):
                raise RedoIntegrityError(f"regular redo {operation.action.value} effect is missing {uri}")
            return ContextObject.from_dict(snapshot["object"]), snapshot

        object_payload = operation.payload.get("context_object")
        desired = ContextObject.from_dict(object_payload) if isinstance(object_payload, dict) else None
        if operation.action in {OperationAction.ADD, OperationAction.UPDATE, OperationAction.MERGE}:
            if desired is None:
                raise RedoIntegrityError("regular object write has no desired object")
            actual, snapshot = required(desired.uri)
            normalized = committer._normalized_regular_object_effect(operation)
            if not isinstance(normalized, dict) or canonical_json(actual.to_dict()) != canonical_json(normalized):
                raise RedoIntegrityError("regular object write did not persist its desired object")
            content = str(operation.payload.get("content", ""))
            if content:
                content_uri = actual.layers.l2_uri or actual.uri
                if dict(snapshot.get("layer_hashes", {}) or {}).get(content_uri) != text_digest(content):
                    raise RedoIntegrityError("regular object write did not persist its desired content")
            return
        if operation.action == OperationAction.SUPERSEDE:
            if not operation.target_uri or desired is None:
                raise RedoIntegrityError("supersede effect is missing an old or replacement URI")
            old, _old_snapshot = required(operation.target_uri)
            new, new_snapshot = required(desired.uri)
            reason = str(operation.payload.get("reason") or operation.payload.get("supersede_reason") or "")
            superseded_at = str(new.metadata.get("superseded_at") or "")
            expected_new = ContextObject.from_dict(desired.to_dict())
            expected_new.lifecycle_state = LifecycleState.ACTIVE
            expected_new.metadata = {
                **expected_new.metadata,
                "supersedes": old.uri,
                "superseded_at": superseded_at,
                "supersede_reason": reason,
            }
            content = str(operation.payload.get("content", ""))
            content_uri = new.layers.l2_uri or new.uri
            if (
                old.lifecycle_state != LifecycleState.OBSOLETE
                or str(old.metadata.get("superseded_by") or "") != new.uri
                or str(old.metadata.get("supersede_reason") or "") != reason
                or not superseded_at
                or str(old.metadata.get("superseded_at") or "") != superseded_at
                or new.lifecycle_state != LifecycleState.ACTIVE
                or str(new.metadata.get("supersedes") or "") != old.uri
                or str(new.metadata.get("supersede_reason") or "") != reason
                or canonical_json(new.to_dict()) != canonical_json(expected_new.to_dict())
                or (
                    content
                    and dict(new_snapshot.get("layer_hashes", {}) or {}).get(content_uri) != text_digest(content)
                )
            ):
                raise RedoIntegrityError("supersede SourceStore effect is incomplete")
            return
        if not operation.target_uri:
            raise RedoIntegrityError(f"regular redo {operation.action.value} has no target URI")
        target, snapshot = required(operation.target_uri)
        if operation.action == OperationAction.DELETE:
            if (
                target.lifecycle_state != LifecycleState.DELETED
                or target.metadata.get("delete_reason") != OperationAction.DELETE.value
            ):
                raise RedoIntegrityError("delete SourceStore effect is not the durable soft-delete state")
            return
        if operation.action == OperationAction.ARCHIVE:
            if (
                target.lifecycle_state != LifecycleState.ARCHIVED
                or target.metadata.get("archive_reason") != operation.payload.get("reason", "")
                or not target.metadata.get("archived_at")
            ):
                raise RedoIntegrityError("archive SourceStore effect is incomplete")
            return
        if operation.action == OperationAction.COMPRESS:
            layer_hashes = dict(snapshot.get("layer_hashes", {}) or {})
            if (
                target.lifecycle_state != LifecycleState.COLD
                or target.metadata.get("compression_reason") != operation.payload.get("reason", "")
                or not target.metadata.get("compressed_at")
                or not target.layers.l0_uri
                or not target.layers.l1_uri
                or not target.layers.l2_uri
                or any(layer_hashes.get(uri) is None for uri in target.layers.to_dict().values() if uri)
            ):
                raise RedoIntegrityError("compress SourceStore effect is incomplete")
            return
        if operation.action == OperationAction.REFRESH_LAYERS:
            layer_hashes = dict(snapshot.get("layer_hashes", {}) or {})
            if (
                not target.layers.l0_uri
                or not target.layers.l1_uri
                or not target.layers.l2_uri
                or any(layer_hashes.get(uri) is None for uri in target.layers.to_dict().values() if uri)
            ):
                raise RedoIntegrityError("layer refresh SourceStore effect is incomplete")
            return
        policy_actions = {
            OperationAction.REWARD,
            OperationAction.PENALIZE,
            OperationAction.COOLDOWN,
            OperationAction.SUPPRESS,
            OperationAction.DISABLE,
        }
        if operation.action in policy_actions and operation.context_type == ContextType.ACTION_POLICY:
            applied = {str(item) for item in target.metadata.get("applied_operation_ids", []) or []}
            if operation.operation_id not in applied:
                raise RedoIntegrityError("action-policy SourceStore effect is missing its operation id")
            return
        if operation.action == OperationAction.DISABLE:
            if (
                target.lifecycle_state != LifecycleState.DELETED
                or target.metadata.get("delete_reason") != OperationAction.DISABLE.value
            ):
                raise RedoIntegrityError("disable SourceStore effect is not durable")
            return
        if operation.action != OperationAction.REINDEX:
            raise RedoIntegrityError(f"unsupported regular redo action: {operation.action.value}")

    @staticmethod
    def _build_regular_relation_manifest(committer, operation: ContextOperation) -> dict:
        """Bind the exact managed relation delta before the Source mutation."""

        expected: list[dict] = []
        previous: list[dict] = []
        desired: ContextObject | None = None
        current: ContextObject | None = None
        if committer.relation_store is not None:
            object_payload = operation.payload.get("context_object")
            desired = ContextObject.from_dict(object_payload) if isinstance(object_payload, dict) else None
            current_uri = operation.target_uri or (desired.uri if desired is not None else None)
            if current_uri:
                try:
                    current = committer.source_store.read_object(str(current_uri))
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                    current = None
            if operation.action in {OperationAction.ADD, OperationAction.UPDATE, OperationAction.MERGE}:
                if current is not None:
                    previous.extend(committer._relation_specs_for_object(current))
                if desired is not None:
                    expected.extend(committer._relation_specs_for_object(desired))
            elif operation.action == OperationAction.SUPERSEDE:
                if desired is not None:
                    try:
                        previous_new = committer.source_store.read_object(desired.uri)
                    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                        previous_new = None
                    if previous_new is not None:
                        previous.extend(committer._relation_specs_for_object(previous_new))
                    expected.extend(committer._relation_specs_for_object(desired))
                if current is not None and desired is not None:
                    relation_metadata = {
                        "tenant_id": desired.tenant_id or current.tenant_id or "default",
                        "owner_user_id": desired.owner_user_id or current.owner_user_id,
                    }
                    expected.extend(
                        [
                            committer._relation_spec(
                                desired.uri,
                                "supersedes",
                                current.uri,
                                relation_metadata,
                            ),
                            committer._relation_spec(
                                current.uri,
                                "superseded_by",
                                desired.uri,
                                relation_metadata,
                            ),
                        ]
                    )
            elif current is not None:
                previous.extend(committer._relation_specs_for_object(current))
                expected.extend(committer._relation_specs_for_object(current))

        expected = committer._unique_relation_specs(expected)
        previous = committer._unique_relation_specs(previous)
        if any(committer._regular_relation_has_canonical_endpoint(spec) for spec in expected):
            raise ValueError(
                "regular operations cannot publish a canonical Source relation; "
                "canonical Source relations require an immutable canonical receipt"
            )
        authority_uri = str(
            (desired.uri if desired is not None else "")
            or (current.uri if current is not None else "")
            or operation.target_uri
            or ""
        )
        previous_keys = {committer._relation_spec_key(spec) for spec in previous}
        serving_expected: list[dict] = []
        for spec in expected:
            eligibility = committer._ordinary_relation_eligibility(
                spec,
                authority_uri=authority_uri,
                authority_object=desired or current,
            )
            if eligibility.allowed:
                serving_expected.append(spec)
                continue
            if committer._action_policy_source_only_relation(desired, spec, eligibility):
                # A retired ActionPolicy still owns its schema-declared
                # evidence relation in Source, but the relation is
                # deliberately absent from the serving manifest.
                continue
            if committer._relation_spec_key(spec) not in previous_keys:
                raise ValueError(f"ordinary relation is not serving-eligible: {eligibility.reason}")
        expected = committer._unique_relation_specs(serving_expected)
        expected_keys = {committer._relation_spec_key(spec) for spec in expected}
        remove = [
            committer._relation_key_payload(spec) for spec in previous if committer._relation_spec_key(spec) not in expected_keys
        ]
        remove = committer._unique_relation_keys(remove)
        core = {
            "schema_version": "regular_relation_manifest_v1",
            "operation_id": operation.operation_id,
            "operation_fingerprint": committer._operation_effect_fingerprint(operation),
            "user_id": operation.user_id,
            "tenant_id": committer._regular_operation_tenant(operation),
            "context_type": operation.context_type.value,
            "target_uri": operation.target_uri,
            "expected": expected,
            "remove": remove,
        }
        return {**core, "fingerprint": stable_hash(core, length=64)}

    @staticmethod
    def _action_policy_source_only_relation(
        committer,
        desired: ContextObject | None,
        spec: dict,
        eligibility: OrdinaryRelationEligibility,
    ) -> bool:
        """Allow only typed retired-policy facts to bypass serving publication."""

        if (
            desired is None
            or desired.context_type != ContextType.ACTION_POLICY
            or desired.lifecycle_state == LifecycleState.ACTIVE
            or str(spec.get("source_uri") or "") != desired.uri
            or eligibility.reason != "source endpoint is not serving"
        ):
            return False
        schema_authority = ContextObject.from_dict(desired.to_dict())
        schema_authority.relations = []
        schema_keys = {committer._relation_spec_key(item) for item in ordinary_relation_specs_for_object(schema_authority)}
        return committer._relation_spec_key(spec) in schema_keys

    @staticmethod
    def _validate_regular_relation_manifest(
        committer,
        operation: ContextOperation,
        manifest: dict | None,
    ) -> None:
        if not isinstance(manifest, dict):
            raise RedoIntegrityError("regular redo entry is missing its relation manifest")
        if manifest.get("schema_version") != "regular_relation_manifest_v1":
            raise RedoIntegrityError("regular redo relation manifest schema is unsupported")
        core = {key: value for key, value in manifest.items() if key != "fingerprint"}
        if manifest.get("fingerprint") != stable_hash(core, length=64):
            raise RedoIntegrityError("regular redo relation manifest fingerprint is corrupt")
        if (
            manifest.get("operation_id") != operation.operation_id
            or manifest.get("operation_fingerprint") != committer._operation_effect_fingerprint(operation)
            or manifest.get("user_id") != operation.user_id
            or manifest.get("tenant_id") != committer._regular_operation_tenant(operation)
            or manifest.get("context_type") != operation.context_type.value
            or manifest.get("target_uri") != operation.target_uri
            or not isinstance(manifest.get("expected"), list)
            or not isinstance(manifest.get("remove"), list)
        ):
            raise RedoIntegrityError("regular redo relation manifest crosses its operation boundary")
        expected = [dict(item) for item in manifest.get("expected", []) if isinstance(item, dict)]
        remove = [dict(item) for item in manifest.get("remove", []) if isinstance(item, dict)]
        if len(expected) != len(manifest.get("expected", [])) or len(remove) != len(manifest.get("remove", [])):
            raise RedoIntegrityError("regular redo relation manifest contains an invalid entry")
        if expected != committer._unique_relation_specs(expected) or remove != committer._unique_relation_keys(remove):
            raise RedoIntegrityError("regular redo relation manifest is not canonical")
        if any(committer._regular_relation_has_canonical_endpoint(spec) for spec in expected):
            raise RedoIntegrityError("regular redo relation manifest crosses the canonical memory boundary")
        expected_keys = {committer._relation_spec_key(spec) for spec in expected}
        if any(committer._relation_spec_key(item) in expected_keys for item in remove):
            raise RedoIntegrityError("regular redo relation manifest removes an expected relation")

    @staticmethod
    def _apply_regular_relation_manifest(
        committer,
        operation: ContextOperation,
        manifest: dict,
    ) -> None:
        committer._validate_regular_relation_manifest(operation, manifest)
        if committer.relation_store is None:
            if manifest.get("expected") or manifest.get("remove"):
                raise RedoIntegrityError("regular relation manifest requires a RelationStore")
            return
        for key in manifest.get("remove", []) or []:
            committer.relation_store.delete_relation(
                str(key["source_uri"]),
                str(key["relation_type"]),
                str(key["target_uri"]),
                tenant_id=str(manifest["tenant_id"]),
            )
        committer._ensure_relation_specs([dict(item) for item in manifest.get("expected", []) or []])
        committer._validate_regular_relation_manifest_effect(manifest)

    @staticmethod
    def _validate_regular_relation_manifest_effect(committer, manifest: dict) -> None:
        if committer.relation_store is None:
            if manifest.get("expected") or manifest.get("remove"):
                raise RedoIntegrityError("regular relation effect has no RelationStore")
            return
        for spec in manifest.get("expected", []) or []:
            relations = committer.relation_store.relations_of(
                str(spec["source_uri"]),
                tenant_id=str(manifest["tenant_id"]),
            )
            actual = {canonical_json(committer._relation_effect_spec(relation)) for relation in relations}
            eligibility = committer._ordinary_relation_eligibility(
                dict(spec),
                authority_uri=str(manifest.get("target_uri") or ""),
            )
            if not eligibility.allowed:
                if canonical_json(spec) in actual:
                    raise RedoIntegrityError("retired ordinary relation remained in RelationStore")
                continue
            if canonical_json(spec) not in actual:
                raise RedoIntegrityError("regular redo RelationStore effect is incomplete")
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
                raise RedoIntegrityError("regular redo RelationStore retained a removed managed relation")

    @staticmethod
    def _relation_spec(
        committer,
        source_uri: str,
        relation_type: str,
        target_uri: str,
        metadata: dict,
        *,
        weight: float = 1.0,
    ) -> dict:
        return {
            "source_uri": source_uri,
            "relation_type": relation_type,
            "target_uri": target_uri,
            "weight": float(weight),
            "metadata": {key: value for key, value in metadata.items() if value is not None},
        }

    @staticmethod
    def _regular_relation_has_canonical_endpoint(committer, spec: dict) -> bool:
        uri = str(spec.get("source_uri") or "")
        if not uri or not uri.startswith("memoryos://"):
            return False
        policy = committer.relation_domain_policy
        if policy is None:
            return False
        if policy.owns_uri(uri):
            return True
        try:
            obj = committer.source_store.read_object(uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return False
        if policy.owns_object(obj):
            return True
        return False

    @staticmethod
    def _ordinary_relation_eligibility(
        committer,
        spec: dict,
        *,
        authority_uri: str = "",
        authority_object: ContextObject | None = None,
    ) -> OrdinaryRelationEligibility:
        tenant_id = str(dict(spec.get("metadata", {}) or {}).get("tenant_id") or committer.tenant_id)
        return ordinary_relation_serving_eligibility(
            spec,
            authority_uri=authority_uri,
            tenant_id=tenant_id,
            source_store=committer.source_store,
            index_store=committer.index_store,
            authority_object=authority_object,
            domain_policy=committer.relation_domain_policy,
            domain_reader=(
                (lambda uri: committer._read_committed_canonical(uri).object)
                if committer.relation_domain_policy is not None
                else None
            ),
            allow_virtual_targets=True,
        )

    @staticmethod
    def _relation_spec_key(committer, spec: dict) -> tuple[str, str, str]:
        return (
            str(spec.get("source_uri") or ""),
            str(spec.get("relation_type") or ""),
            str(spec.get("target_uri") or ""),
        )

    @staticmethod
    def _relation_key_payload(committer, spec: dict) -> dict:
        source_uri, relation_type, target_uri = committer._relation_spec_key(spec)
        return {
            "source_uri": source_uri,
            "relation_type": relation_type,
            "target_uri": target_uri,
        }

    @staticmethod
    def _unique_relation_specs(committer, specs: list[dict]) -> list[dict]:
        unique = {canonical_json(spec): spec for spec in specs}
        return [unique[key] for key in sorted(unique)]

    @staticmethod
    def _unique_relation_keys(committer, keys: list[dict]) -> list[dict]:
        unique = {canonical_json(key): key for key in keys}
        return [unique[key] for key in sorted(unique)]

    @staticmethod
    def _expected_regular_relation_specs(committer, operation: ContextOperation) -> list[dict]:
        if committer.relation_store is None:
            return []
        specs: list[dict] = []
        object_payload = operation.payload.get("context_object")
        if operation.action in {OperationAction.ADD, OperationAction.UPDATE, OperationAction.MERGE}:
            if isinstance(object_payload, dict):
                try:
                    obj = committer.source_store.read_object(str(object_payload["uri"]))
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                    obj = ContextObject.from_dict(object_payload)
                specs.extend(committer._relation_specs_for_object(obj))
        elif operation.action == OperationAction.SUPERSEDE:
            if operation.target_uri and isinstance(object_payload, dict):
                old_obj = committer.source_store.read_object(operation.target_uri)
                try:
                    new_obj = committer.source_store.read_object(str(object_payload["uri"]))
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                    new_obj = ContextObject.from_dict(object_payload)
                specs.extend(committer._relation_specs_for_object(new_obj))
                metadata = {
                    "tenant_id": new_obj.tenant_id or old_obj.tenant_id or "default",
                    "owner_user_id": new_obj.owner_user_id or old_obj.owner_user_id,
                }
                specs.extend(
                    [
                        {
                            "source_uri": new_obj.uri,
                            "relation_type": "supersedes",
                            "target_uri": old_obj.uri,
                            "weight": 1.0,
                            "metadata": {key: value for key, value in metadata.items() if value is not None},
                        },
                        {
                            "source_uri": old_obj.uri,
                            "relation_type": "superseded_by",
                            "target_uri": new_obj.uri,
                            "weight": 1.0,
                            "metadata": {key: value for key, value in metadata.items() if value is not None},
                        },
                    ]
                )
        elif (
            operation.context_type == ContextType.ACTION_POLICY
            and operation.target_uri
            and operation.action
            in {
                OperationAction.REWARD,
                OperationAction.PENALIZE,
                OperationAction.COOLDOWN,
                OperationAction.SUPPRESS,
                OperationAction.DISABLE,
            }
        ):
            specs.extend(committer._relation_specs_for_object(committer.source_store.read_object(operation.target_uri)))
        unique = {canonical_json(spec): spec for spec in specs}
        return [unique[key] for key in sorted(unique)]

    @staticmethod
    def _restore_regular_relation_effect(committer, operation: ContextOperation, source_effect: dict) -> None:
        expected = committer._expected_regular_relation_specs(operation)
        if source_effect.get("relations") != expected:
            raise RedoIntegrityError("regular redo relation effect does not match its operation")
        committer._ensure_relation_specs(expected)

    @staticmethod
    def _validate_regular_relation_postcondition(committer, expected: list[dict]) -> None:
        if committer.relation_store is None:
            if expected:
                raise RedoIntegrityError("regular redo relation effect has no RelationStore")
            return
        for spec in expected:
            tenant_id = str(dict(spec.get("metadata", {}) or {}).get("tenant_id") or committer.tenant_id)
            relations = committer.relation_store.relations_of(
                str(spec["source_uri"]),
                tenant_id=tenant_id,
            )
            actual = {canonical_json(committer._relation_effect_spec(relation)) for relation in relations}
            eligibility = committer._ordinary_relation_eligibility(spec)
            if not eligibility.allowed:
                if canonical_json(spec) in actual:
                    raise RedoIntegrityError("retired ordinary relation remained in RelationStore")
                continue
            if canonical_json(spec) not in actual:
                raise RedoIntegrityError("regular redo RelationStore effect is incomplete")

    @staticmethod
    def _regular_source_effect_uris(committer, operation: ContextOperation) -> list[str]:
        uris: list[str] = []
        if operation.target_uri:
            uris.append(str(operation.target_uri))
        object_payload = operation.payload.get("context_object")
        if isinstance(object_payload, dict) and object_payload.get("uri"):
            uris.append(str(object_payload["uri"]))
        return list(dict.fromkeys(uris))

    @staticmethod
    def _regular_operation_tenant(committer, operation: ContextOperation) -> str:
        if not committer._operation_matches_bound_tenant(operation):
            raise ValueError("regular operation tenant does not match bound tenant")
        return committer.tenant_id
