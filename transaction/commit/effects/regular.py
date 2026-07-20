"""普通操作的副作用快照、关系清单与恢复校验组件。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from foundation.ids import stable_hash
from foundation.integrity import canonical_json, text_digest
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_uri import ContextURI
from infrastructure.store.model.context.lifecycle import LifecycleState
from transaction.commit.control import RedoIntegrityError
from transaction.commit.domain_protocols import RelationEligibility
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_action import OperationAction

if TYPE_CHECKING:
    from transaction.commit.host import OperationTransactionHost


class RegularEffectExecutor:
    """负责普通操作的副作用快照、关系清单与恢复校验。"""

    def _capture_regular_source_effect(
        self: OperationTransactionHost,
        operation: ContextOperation,
        relation_manifest: dict | None = None,
    ) -> dict:
        uris = self._regular_source_effect_uris(operation)
        snapshots = []
        for uri in uris:
            try:
                obj = self.source_store.read_object(uri)
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
                    content = self.source_store.read_content(str(layer_uri))
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
                else self._expected_regular_relation_specs(operation)
            ),
            "relation_manifest_fingerprint": (
                str(relation_manifest.get("fingerprint") or "") if isinstance(relation_manifest, dict) else ""
            ),
        }
        return {**core, "fingerprint": stable_hash(core, length=64)}

    def _validate_regular_recovery_effect(
        self: OperationTransactionHost,
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
            or list(source_effect.get("uris", []) or []) != self._regular_source_effect_uris(operation)
        ):
            raise RedoIntegrityError("regular redo SourceStore effect is bound to another operation")
        actual = self._capture_regular_source_effect(operation, relation_manifest)
        if actual.get("fingerprint") != source_effect.get("fingerprint"):
            raise RedoIntegrityError("regular redo SourceStore effect does not match durable state")
        expected_tenant = self._regular_operation_tenant(operation)
        self._validate_regular_action_postcondition(operation, actual)
        if relation_manifest is not None:
            self._validate_regular_relation_manifest(operation, relation_manifest)
        elif self.relation_store is not None:
            raise RedoIntegrityError("regular redo entry is missing its relation manifest")
        expected_relations = (
            list(relation_manifest.get("expected", []) or [])
            if isinstance(relation_manifest, dict)
            else self._expected_regular_relation_specs(operation)
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
                self._validate_regular_relation_manifest_effect(relation_manifest)
            else:
                self._validate_regular_relation_postcondition(expected_relations)

    def _validate_and_restore_regular_recovery_effect(
        self: OperationTransactionHost,
        user_id: str,
        operation: ContextOperation,
        source_effect: dict | None,
        relation_manifest: dict | None,
    ) -> None:
        self._validate_regular_recovery_effect(
            user_id,
            operation,
            source_effect,
            require_relation_presence=False,
            relation_manifest=relation_manifest,
        )
        if isinstance(relation_manifest, dict):
            self._apply_regular_relation_manifest(operation, relation_manifest)
        else:
            assert isinstance(source_effect, dict)
            self._restore_regular_relation_effect(operation, source_effect)
        self._validate_regular_recovery_effect(
            user_id,
            operation,
            source_effect,
            relation_manifest=relation_manifest,
        )

    def _validate_regular_action_postcondition(
        self: OperationTransactionHost,
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
            normalized = self._normalized_regular_object_effect(operation)
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
        if self._validate_domain_postcondition(operation, effect):
            return
        if operation.action != OperationAction.REINDEX:
            raise RedoIntegrityError(f"unsupported regular redo action: {operation.action.value}")

    def _build_regular_relation_manifest(self: OperationTransactionHost, operation: ContextOperation) -> dict:
        """修改 Source 前固定本次事务管理的精确关系差异。"""

        expected: list[dict] = []
        previous: list[dict] = []
        desired: ContextObject | None = None
        current: ContextObject | None = None
        if self.relation_store is not None:
            object_payload = operation.payload.get("context_object")
            desired = ContextObject.from_dict(object_payload) if isinstance(object_payload, dict) else None
            current_uri = operation.target_uri or (desired.uri if desired is not None else None)
            if current_uri:
                try:
                    current = self.source_store.read_object(str(current_uri))
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                    current = None
            if operation.action in {OperationAction.ADD, OperationAction.UPDATE, OperationAction.MERGE}:
                if current is not None:
                    previous.extend(self._relation_specs_for_object(current))
                if desired is not None:
                    expected.extend(self._relation_specs_for_object(desired))
            elif operation.action == OperationAction.SUPERSEDE:
                if desired is not None:
                    try:
                        previous_new = self.source_store.read_object(desired.uri)
                    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                        previous_new = None
                    if previous_new is not None:
                        previous.extend(self._relation_specs_for_object(previous_new))
                    expected.extend(self._relation_specs_for_object(desired))
                if current is not None and desired is not None:
                    relation_metadata = {
                        "tenant_id": desired.tenant_id or current.tenant_id or "default",
                        "owner_user_id": desired.owner_user_id or current.owner_user_id,
                    }
                    expected.extend(
                        [
                            self._relation_spec(
                                desired.uri,
                                "supersedes",
                                current.uri,
                                relation_metadata,
                            ),
                            self._relation_spec(
                                current.uri,
                                "superseded_by",
                                desired.uri,
                                relation_metadata,
                            ),
                        ]
                    )
            elif current is not None:
                previous.extend(self._relation_specs_for_object(current))
                expected.extend(self._relation_specs_for_object(current))

        expected = self._unique_relation_specs(expected)
        previous = self._unique_relation_specs(previous)
        authority_uri = str(
            (desired.uri if desired is not None else "")
            or (current.uri if current is not None else "")
            or operation.target_uri
            or ""
        )
        previous_keys = {self._relation_spec_key(spec) for spec in previous}
        serving_expected: list[dict] = []
        for spec in expected:
            eligibility = self._ordinary_relation_eligibility(
                spec,
                authority_uri=authority_uri,
                authority_object=desired or current,
            )
            if eligibility.allowed:
                serving_expected.append(spec)
                continue
            if self._domain_allows_source_only_relation(desired, spec, eligibility):
                continue
            if self._relation_spec_key(spec) not in previous_keys:
                raise ValueError(f"ordinary relation is not serving-eligible: {eligibility.reason}")
        expected = self._unique_relation_specs(serving_expected)
        expected_keys = {self._relation_spec_key(spec) for spec in expected}
        remove = [
            self._relation_key_payload(spec) for spec in previous if self._relation_spec_key(spec) not in expected_keys
        ]
        remove = self._unique_relation_keys(remove)
        core = {
            "schema_version": "regular_relation_manifest_v1",
            "operation_id": operation.operation_id,
            "operation_fingerprint": self._operation_effect_fingerprint(operation),
            "user_id": operation.user_id,
            "tenant_id": self._regular_operation_tenant(operation),
            "context_type": operation.context_type.value,
            "target_uri": operation.target_uri,
            "expected": expected,
            "remove": remove,
        }
        return {**core, "fingerprint": stable_hash(core, length=64)}

    def _validate_regular_relation_manifest(
        self: OperationTransactionHost,
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
            or manifest.get("operation_fingerprint") != self._operation_effect_fingerprint(operation)
            or manifest.get("user_id") != operation.user_id
            or manifest.get("tenant_id") != self._regular_operation_tenant(operation)
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
        if expected != self._unique_relation_specs(expected) or remove != self._unique_relation_keys(remove):
            raise RedoIntegrityError("regular redo relation manifest is not normalized")
        expected_keys = {self._relation_spec_key(spec) for spec in expected}
        if any(self._relation_spec_key(item) in expected_keys for item in remove):
            raise RedoIntegrityError("regular redo relation manifest removes an expected relation")

    def _apply_regular_relation_manifest(
        self: OperationTransactionHost,
        operation: ContextOperation,
        manifest: dict,
    ) -> None:
        self._validate_regular_relation_manifest(operation, manifest)
        if self.relation_store is None:
            if manifest.get("expected") or manifest.get("remove"):
                raise RedoIntegrityError("regular relation manifest requires a RelationStore")
            return
        for key in manifest.get("remove", []) or []:
            self.relation_store.delete_relation(
                str(key["source_uri"]),
                str(key["relation_type"]),
                str(key["target_uri"]),
                tenant_id=str(manifest["tenant_id"]),
            )
        self._ensure_relation_specs([dict(item) for item in manifest.get("expected", []) or []])
        self._validate_regular_relation_manifest_effect(manifest)

    def _validate_regular_relation_manifest_effect(self: OperationTransactionHost, manifest: dict) -> None:
        if self.relation_store is None:
            if manifest.get("expected") or manifest.get("remove"):
                raise RedoIntegrityError("regular relation effect has no RelationStore")
            return
        for spec in manifest.get("expected", []) or []:
            relations = self.relation_store.relations_of(
                str(spec["source_uri"]),
                tenant_id=str(manifest["tenant_id"]),
            )
            actual = {canonical_json(self._relation_effect_spec(relation)) for relation in relations}
            eligibility = self._ordinary_relation_eligibility(
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
                for relation in self.relation_store.relations_of(
                    str(key["source_uri"]),
                    tenant_id=str(manifest["tenant_id"]),
                )
            ):
                raise RedoIntegrityError("regular redo RelationStore retained a removed managed relation")

    def _relation_spec(
        self: OperationTransactionHost,
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

    def _ordinary_relation_eligibility(
        self: OperationTransactionHost,
        spec: dict,
        *,
        authority_uri: str = "",
        authority_object: ContextObject | None = None,
    ) -> RelationEligibility:
        tenant_id = str(dict(spec.get("metadata", {}) or {}).get("tenant_id") or self.tenant_id)
        if self.context_effects is None:
            raise RuntimeError("relation validation requires an injected ContextOperationEffects")
        return self.context_effects.relation_eligibility(
            spec,
            authority_uri=authority_uri,
            tenant_id=tenant_id,
            source_store=self.source_store,
            index_store=self.index_store,
            authority_object=authority_object,
        )

    def _relation_spec_key(self: OperationTransactionHost, spec: dict) -> tuple[str, str, str]:
        return (
            str(spec.get("source_uri") or ""),
            str(spec.get("relation_type") or ""),
            str(spec.get("target_uri") or ""),
        )

    def _relation_key_payload(self: OperationTransactionHost, spec: dict) -> dict:
        source_uri, relation_type, target_uri = self._relation_spec_key(spec)
        return {
            "source_uri": source_uri,
            "relation_type": relation_type,
            "target_uri": target_uri,
        }

    def _unique_relation_specs(self: OperationTransactionHost, specs: list[dict]) -> list[dict]:
        unique = {canonical_json(spec): spec for spec in specs}
        return [unique[key] for key in sorted(unique)]

    def _unique_relation_keys(self: OperationTransactionHost, keys: list[dict]) -> list[dict]:
        unique = {canonical_json(key): key for key in keys}
        return [unique[key] for key in sorted(unique)]

    def _expected_regular_relation_specs(self: OperationTransactionHost, operation: ContextOperation) -> list[dict]:
        if self.relation_store is None:
            return []
        specs: list[dict] = []
        object_payload = operation.payload.get("context_object")
        if operation.action in {OperationAction.ADD, OperationAction.UPDATE, OperationAction.MERGE}:
            if isinstance(object_payload, dict):
                try:
                    obj = self.source_store.read_object(str(object_payload["uri"]))
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                    obj = ContextObject.from_dict(object_payload)
                specs.extend(self._relation_specs_for_object(obj))
        elif operation.action == OperationAction.SUPERSEDE:
            if operation.target_uri and isinstance(object_payload, dict):
                old_obj = self.source_store.read_object(operation.target_uri)
                try:
                    new_obj = self.source_store.read_object(str(object_payload["uri"]))
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                    new_obj = ContextObject.from_dict(object_payload)
                specs.extend(self._relation_specs_for_object(new_obj))
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
        elif operation.target_uri and self._domain_handler_for(operation) is not None:
            specs.extend(self._relation_specs_for_object(self.source_store.read_object(operation.target_uri)))
        unique = {canonical_json(spec): spec for spec in specs}
        return [unique[key] for key in sorted(unique)]

    def _restore_regular_relation_effect(
        self: OperationTransactionHost, operation: ContextOperation, source_effect: dict
    ) -> None:
        expected = self._expected_regular_relation_specs(operation)
        if source_effect.get("relations") != expected:
            raise RedoIntegrityError("regular redo relation effect does not match its operation")
        self._ensure_relation_specs(expected)

    def _validate_regular_relation_postcondition(self: OperationTransactionHost, expected: list[dict]) -> None:
        if self.relation_store is None:
            if expected:
                raise RedoIntegrityError("regular redo relation effect has no RelationStore")
            return
        for spec in expected:
            tenant_id = str(dict(spec.get("metadata", {}) or {}).get("tenant_id") or self.tenant_id)
            relations = self.relation_store.relations_of(
                str(spec["source_uri"]),
                tenant_id=tenant_id,
            )
            actual = {canonical_json(self._relation_effect_spec(relation)) for relation in relations}
            eligibility = self._ordinary_relation_eligibility(spec)
            if not eligibility.allowed:
                if canonical_json(spec) in actual:
                    raise RedoIntegrityError("retired ordinary relation remained in RelationStore")
                continue
            if canonical_json(spec) not in actual:
                raise RedoIntegrityError("regular redo RelationStore effect is incomplete")

    def _regular_source_effect_uris(self: OperationTransactionHost, operation: ContextOperation) -> list[str]:
        uris: list[str] = []
        if operation.target_uri:
            uris.append(str(operation.target_uri))
        object_payload = operation.payload.get("context_object")
        if isinstance(object_payload, dict) and object_payload.get("uri"):
            uris.append(str(object_payload["uri"]))
        return list(dict.fromkeys(uris))

    def _regular_operation_tenant(self: OperationTransactionHost, operation: ContextOperation) -> str:
        if not self._operation_matches_bound_tenant(operation):
            raise ValueError("regular operation tenant does not match bound tenant")
        return self.tenant_id
