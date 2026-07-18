"""Implementation component for StoreEffectWriter.

The public OperationCommitter delegates explicitly to this component so fault
injection hooks remain available on the facade.
"""

from __future__ import annotations

from memoryos.contextdb.layers.layer_refresher import LayerRefresher
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.ordinary_relations import (
    ordinary_relation_specs_for_object,
)
from memoryos.core.clock import utc_now
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class StoreEffectWriter:
    """Own generic Source, Index, and Relation writes for a commit."""

    @staticmethod
    def _coalesce_non_policy_operations(committer, operations: list[ContextOperation]) -> list[ContextOperation]:
        policy_actions = {
            OperationAction.REWARD,
            OperationAction.PENALIZE,
            OperationAction.COOLDOWN,
            OperationAction.SUPPRESS,
            OperationAction.DISABLE,
        }
        policy_ops = [operation for operation in operations if operation.action in policy_actions]
        other_ops = [operation for operation in operations if operation.action not in policy_actions]
        return [*committer.coalescer.coalesce(other_ops), *policy_ops]

    @staticmethod
    def _apply_source(committer, operation: ContextOperation) -> None:
        if operation.action == OperationAction.SUPERSEDE:
            committer._apply_supersede_source(operation)
            return
        if operation.action in {OperationAction.ADD, OperationAction.UPDATE, OperationAction.MERGE}:
            object_payload = operation.payload.get("context_object")
            if isinstance(object_payload, dict):
                obj = ContextObject.from_dict(object_payload)
                obj = committer._materialize_action_policy_source_relations(obj)
                content = str(operation.payload.get("content", ""))
                committer.source_store.write_object(obj, content=content)
                if content:
                    LayerRefresher(committer.source_store).refresh(obj, content)
                    operation.payload["context_object"] = obj.to_dict()
                committer._apply_relations(obj, operation)
            return
        if (
            operation.action
            in {
                OperationAction.REWARD,
                OperationAction.PENALIZE,
                OperationAction.COOLDOWN,
                OperationAction.SUPPRESS,
                OperationAction.DISABLE,
            }
            and operation.target_uri
        ):
            if operation.context_type == ContextType.ACTION_POLICY:
                policy = committer._read_action_policy(operation.target_uri)
                policy = committer._apply_action_policy_mutation(policy, operation)
                committer._write_action_policy(policy)
            elif operation.action == OperationAction.DISABLE:
                committer.source_store.soft_delete(operation.target_uri, operation.action.value)
            return
        if operation.action == OperationAction.COMPRESS and operation.target_uri:
            obj = committer.source_store.read_object(operation.target_uri)
            content = committer._read_content_or_empty(operation.target_uri)
            LayerRefresher(committer.source_store).refresh(
                obj, content, bullets=[operation.payload.get("reason", "compressed")]
            )
            obj.lifecycle_state = LifecycleState.COLD
            obj.metadata = {
                **obj.metadata,
                "compressed_at": utc_now(),
                "compression_reason": operation.payload.get("reason", ""),
            }
            committer.source_store.write_object(obj)
            return
        if operation.action == OperationAction.REFRESH_LAYERS and operation.target_uri:
            obj = committer.source_store.read_object(operation.target_uri)
            content = committer._read_content_or_empty(operation.target_uri)
            LayerRefresher(committer.source_store).refresh(obj, content)
            return
        if operation.action == OperationAction.ARCHIVE and operation.target_uri:
            obj = committer.source_store.read_object(operation.target_uri)
            obj.lifecycle_state = LifecycleState.ARCHIVED
            obj.metadata = {
                **obj.metadata,
                "archived_at": utc_now(),
                "archive_reason": operation.payload.get("reason", ""),
            }
            content = committer._read_content_or_empty(operation.target_uri)
            committer.source_store.write_object(obj, content=content)
            return
        if operation.action == OperationAction.DELETE and operation.target_uri:
            committer.source_store.soft_delete(operation.target_uri, operation.action.value)
            return

    @staticmethod
    def _apply_index(committer, operation: ContextOperation) -> None:
        if operation.action == OperationAction.SUPERSEDE:
            committer._apply_supersede_index(operation)
            return
        if operation.action in {OperationAction.ADD, OperationAction.UPDATE, OperationAction.MERGE}:
            object_payload = operation.payload.get("context_object")
            if isinstance(object_payload, dict):
                obj = ContextObject.from_dict(object_payload)
                committer.index_store.upsert_index(
                    obj,
                    content=str(operation.payload.get("content", "")),
                    tenant_id=committer.tenant_id,
                )
            return
        if operation.action == OperationAction.DELETE and operation.target_uri:
            if committer._delete_tombstone_ids(operation):
                # The durable projection worker owns Catalog/FTS/Vector/Path/
                # Relation cleanup.  Synchronously deleting only SQLite here
                # would make the Source transaction look complete while
                # external derived state remained searchable.
                return
            committer.index_store.delete_index(
                operation.target_uri,
                tenant_id=committer.tenant_id,
            )
            return
        if operation.target_uri and operation.action in {
            OperationAction.REWARD,
            OperationAction.PENALIZE,
            OperationAction.COOLDOWN,
            OperationAction.SUPPRESS,
            OperationAction.DISABLE,
            OperationAction.COMPRESS,
            OperationAction.REFRESH_LAYERS,
            OperationAction.ARCHIVE,
            OperationAction.REINDEX,
        }:
            if operation.action == OperationAction.DISABLE and operation.context_type != ContextType.ACTION_POLICY:
                committer.index_store.delete_index(
                    operation.target_uri,
                    tenant_id=committer.tenant_id,
                )
                return
            obj = committer.source_store.read_object(operation.target_uri)
            committer.index_store.upsert_index(
                obj,
                content=committer._read_content_or_empty(operation.target_uri),
                tenant_id=committer.tenant_id,
            )

    @staticmethod
    def _apply_supersede_source(committer, operation: ContextOperation) -> None:
        if not operation.target_uri:
            return
        object_payload = operation.payload.get("context_object")
        if not isinstance(object_payload, dict):
            return
        old_obj = committer.source_store.read_object(operation.target_uri)
        old_content = committer._read_content_or_empty(operation.target_uri)
        new_obj = ContextObject.from_dict(object_payload)
        new_obj.lifecycle_state = LifecycleState.ACTIVE
        superseded_at = utc_now()
        reason = str(operation.payload.get("reason") or operation.payload.get("supersede_reason") or "")
        old_obj.lifecycle_state = LifecycleState.OBSOLETE
        old_obj.metadata = {
            **old_obj.metadata,
            "superseded_at": superseded_at,
            "superseded_by": new_obj.uri,
            "supersede_reason": reason,
        }
        new_obj.metadata = {
            **new_obj.metadata,
            "supersedes": old_obj.uri,
            "superseded_at": superseded_at,
            "supersede_reason": reason,
        }
        committer.source_store.write_object(old_obj, content=old_content)
        committer.source_store.write_object(new_obj, content=str(operation.payload.get("content", "")))
        committer._apply_relations(new_obj, operation)
        committer._add_supersede_relations(old_obj, new_obj)

    @staticmethod
    def _apply_supersede_index(committer, operation: ContextOperation) -> None:
        if not operation.target_uri:
            return
        old_obj = committer.source_store.read_object(operation.target_uri)
        committer.index_store.upsert_index(
            old_obj,
            content=committer._read_content_or_empty(operation.target_uri),
            tenant_id=committer.tenant_id,
        )
        object_payload = operation.payload.get("context_object")
        if isinstance(object_payload, dict):
            new_uri = object_payload.get("uri")
            if not new_uri:
                return
            new_obj = committer.source_store.read_object(str(new_uri))
            committer.index_store.upsert_index(
                new_obj,
                content=str(operation.payload.get("content", "")),
                tenant_id=committer.tenant_id,
            )

    @staticmethod
    def _add_supersede_relations(committer, old_obj: ContextObject, new_obj: ContextObject) -> None:
        metadata = {
            "tenant_id": new_obj.tenant_id or old_obj.tenant_id or "default",
            "owner_user_id": new_obj.owner_user_id or old_obj.owner_user_id,
        }
        committer._add_relation(new_obj.uri, "supersedes", old_obj.uri, metadata)
        committer._add_relation(old_obj.uri, "superseded_by", new_obj.uri, metadata)

    @staticmethod
    def _apply_relations(committer, obj: ContextObject, operation: ContextOperation) -> None:
        del operation
        if committer.relation_store is None:
            return
        committer._ensure_relation_specs(committer._relation_specs_for_object(obj))

    @staticmethod
    def _relation_specs_for_object(committer, obj: ContextObject) -> list[dict]:
        return ordinary_relation_specs_for_object(obj)

    @staticmethod
    def _add_relation(committer, source_uri: str, relation_type: str, target_uri: str, metadata: dict) -> None:
        if committer.relation_store is None or not target_uri:
            return
        committer._ensure_relation_specs(
            [
                {
                    "source_uri": source_uri,
                    "relation_type": relation_type,
                    "target_uri": target_uri,
                    "weight": 1.0,
                    "metadata": {key: value for key, value in metadata.items() if value is not None},
                }
            ]
        )

    @staticmethod
    def _ensure_relation_specs(committer, specs: list[dict]) -> None:
        if committer.relation_store is None:
            return
        for spec in specs:
            tenant_id = str(dict(spec.get("metadata", {}) or {}).get("tenant_id") or committer.tenant_id)
            existing = committer.relation_store.relations_of(
                str(spec["source_uri"]),
                tenant_id=tenant_id,
            )
            matching_key = next(
                (
                    relation
                    for relation in existing
                    if relation.source_uri == spec["source_uri"]
                    and relation.relation_type == spec["relation_type"]
                    and relation.target_uri == spec["target_uri"]
                ),
                None,
            )
            eligibility = committer._ordinary_relation_eligibility(spec)
            if not eligibility.allowed:
                if matching_key is not None:
                    committer.relation_store.delete_relation(
                        matching_key.source_uri,
                        matching_key.relation_type,
                        matching_key.target_uri,
                        tenant_id=tenant_id,
                    )
                continue
            if matching_key is not None and committer._relation_effect_spec(matching_key) == spec:
                continue
            if matching_key is not None:
                committer.relation_store.delete_relation(
                    matching_key.source_uri,
                    matching_key.relation_type,
                    matching_key.target_uri,
                    tenant_id=tenant_id,
                )
            committer.relation_store.add_relation(
                ContextRelation(
                    source_uri=str(spec["source_uri"]),
                    relation_type=str(spec["relation_type"]),
                    target_uri=str(spec["target_uri"]),
                    weight=float(spec.get("weight", 1.0)),
                    metadata=dict(spec.get("metadata", {}) or {}),
                ),
                tenant_id=tenant_id,
            )

    @staticmethod
    def _relation_effect_spec(committer, relation: ContextRelation) -> dict:
        return {
            "source_uri": relation.source_uri,
            "relation_type": relation.relation_type,
            "target_uri": relation.target_uri,
            "weight": float(relation.weight),
            "metadata": dict(relation.metadata),
        }

    @staticmethod
    def _read_content_or_empty(committer, uri: str) -> str:
        try:
            return committer.source_store.read_content(uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return ""
