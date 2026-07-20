"""把已经通过事务校验的普通对象副作用写入 Source、Index 和 Relation。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from foundation.clock import utc_now
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_relation import ContextRelation
from infrastructure.store.model.context.lifecycle import LifecycleState
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_action import OperationAction

if TYPE_CHECKING:
    from transaction.commit.host import OperationTransactionHost


class StoreEffectWriter:
    """负责一次提交中的通用 Source、Index 和 Relation 写入。"""

    def _apply_source(self: OperationTransactionHost, operation: ContextOperation) -> None:
        if operation.action == OperationAction.SUPERSEDE:
            self._apply_supersede_source(operation)
            return
        if self._apply_domain_source(operation):
            return
        if operation.action in {OperationAction.ADD, OperationAction.UPDATE, OperationAction.MERGE}:
            object_payload = operation.payload.get("context_object")
            if isinstance(object_payload, dict):
                obj = ContextObject.from_dict(object_payload)
                obj = self._materialize_domain_object(obj)
                content = str(operation.payload.get("content", ""))
                self.source_store.write_object(obj, content=content)
                if content and self.context_effects is not None:
                    obj = self._refresh_context_layers(obj, content)
                    operation.payload["context_object"] = obj.to_dict()
                self._apply_relations(obj, operation)
            return
        if operation.action == OperationAction.COMPRESS and operation.target_uri:
            obj = self.source_store.read_object(operation.target_uri)
            content = self._read_content_or_empty(operation.target_uri)
            obj = self._refresh_context_layers(
                obj,
                content,
                bullets=[operation.payload.get("reason", "compressed")],
            )
            obj.lifecycle_state = LifecycleState.COLD
            obj.metadata = {
                **obj.metadata,
                "compressed_at": utc_now(),
                "compression_reason": operation.payload.get("reason", ""),
            }
            self.source_store.write_object(obj)
            return
        if operation.action == OperationAction.REFRESH_LAYERS and operation.target_uri:
            obj = self.source_store.read_object(operation.target_uri)
            content = self._read_content_or_empty(operation.target_uri)
            self._refresh_context_layers(obj, content)
            return
        if operation.action == OperationAction.ARCHIVE and operation.target_uri:
            obj = self.source_store.read_object(operation.target_uri)
            obj.lifecycle_state = LifecycleState.ARCHIVED
            obj.metadata = {
                **obj.metadata,
                "archived_at": utc_now(),
                "archive_reason": operation.payload.get("reason", ""),
            }
            content = self._read_content_or_empty(operation.target_uri)
            self.source_store.write_object(obj, content=content)
            return
        if operation.action == OperationAction.DELETE and operation.target_uri:
            self.source_store.soft_delete(operation.target_uri, operation.action.value)
            return

    def _apply_index(self: OperationTransactionHost, operation: ContextOperation) -> None:
        if operation.action == OperationAction.SUPERSEDE:
            self._apply_supersede_index(operation)
            return
        if operation.action in {OperationAction.ADD, OperationAction.UPDATE, OperationAction.MERGE}:
            object_payload = operation.payload.get("context_object")
            if isinstance(object_payload, dict):
                obj = ContextObject.from_dict(object_payload)
                self.index_store.upsert_index(
                    obj,
                    content=str(operation.payload.get("content", "")),
                    tenant_id=self.tenant_id,
                )
            return
        if operation.action == OperationAction.DELETE and operation.target_uri:
            if self._delete_tombstone_ids(operation):
                # Catalog、FTS、向量、路径与关系清理由耐久投影任务统一负责。
                # 这里只同步删除 SQLite 会让 Source 事务看起来已经完成，
                # 但其他派生索引仍可能继续召回被删除对象。
                return
            self.index_store.delete_index(
                operation.target_uri,
                tenant_id=self.tenant_id,
            )
            return
        if operation.target_uri and self._domain_handler_for(operation) is not None:
            obj = self.source_store.read_object(operation.target_uri)
            self.index_store.upsert_index(
                obj,
                content=self._read_content_or_empty(operation.target_uri),
                tenant_id=self.tenant_id,
            )
            return
        if operation.target_uri and operation.action in {
            OperationAction.COMPRESS,
            OperationAction.REFRESH_LAYERS,
            OperationAction.ARCHIVE,
            OperationAction.REINDEX,
        }:
            obj = self.source_store.read_object(operation.target_uri)
            self.index_store.upsert_index(
                obj,
                content=self._read_content_or_empty(operation.target_uri),
                tenant_id=self.tenant_id,
            )

    def _apply_supersede_source(self: OperationTransactionHost, operation: ContextOperation) -> None:
        if not operation.target_uri:
            return
        object_payload = operation.payload.get("context_object")
        if not isinstance(object_payload, dict):
            return
        old_obj = self.source_store.read_object(operation.target_uri)
        old_content = self._read_content_or_empty(operation.target_uri)
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
        self.source_store.write_object(old_obj, content=old_content)
        self.source_store.write_object(new_obj, content=str(operation.payload.get("content", "")))
        self._apply_relations(new_obj, operation)
        self._add_supersede_relations(old_obj, new_obj)

    def _apply_supersede_index(self: OperationTransactionHost, operation: ContextOperation) -> None:
        if not operation.target_uri:
            return
        old_obj = self.source_store.read_object(operation.target_uri)
        self.index_store.upsert_index(
            old_obj,
            content=self._read_content_or_empty(operation.target_uri),
            tenant_id=self.tenant_id,
        )
        object_payload = operation.payload.get("context_object")
        if isinstance(object_payload, dict):
            new_uri = object_payload.get("uri")
            if not new_uri:
                return
            new_obj = self.source_store.read_object(str(new_uri))
            self.index_store.upsert_index(
                new_obj,
                content=str(operation.payload.get("content", "")),
                tenant_id=self.tenant_id,
            )

    def _add_supersede_relations(
        self: OperationTransactionHost, old_obj: ContextObject, new_obj: ContextObject
    ) -> None:
        metadata = {
            "tenant_id": new_obj.tenant_id or old_obj.tenant_id or "default",
            "owner_user_id": new_obj.owner_user_id or old_obj.owner_user_id,
        }
        self._add_relation(new_obj.uri, "supersedes", old_obj.uri, metadata)
        self._add_relation(old_obj.uri, "superseded_by", new_obj.uri, metadata)

    def _apply_relations(self: OperationTransactionHost, obj: ContextObject, operation: ContextOperation) -> None:
        del operation
        if self.relation_store is None:
            return
        self._ensure_relation_specs(self._relation_specs_for_object(obj))

    def _relation_specs_for_object(self: OperationTransactionHost, obj: ContextObject) -> list[dict]:
        if self.context_effects is None:
            if self.relation_store is None:
                return []
            raise RuntimeError("relation projection requires an injected ContextOperationEffects")
        return self.context_effects.relation_specs_for_object(obj)

    def _add_relation(
        self: OperationTransactionHost, source_uri: str, relation_type: str, target_uri: str, metadata: dict
    ) -> None:
        if self.relation_store is None or not target_uri:
            return
        self._ensure_relation_specs(
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

    def _ensure_relation_specs(self: OperationTransactionHost, specs: list[dict]) -> None:
        if self.relation_store is None:
            return
        for spec in specs:
            tenant_id = str(dict(spec.get("metadata", {}) or {}).get("tenant_id") or self.tenant_id)
            existing = self.relation_store.relations_of(
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
            eligibility = self._ordinary_relation_eligibility(spec)
            if not eligibility.allowed:
                if matching_key is not None:
                    self.relation_store.delete_relation(
                        matching_key.source_uri,
                        matching_key.relation_type,
                        matching_key.target_uri,
                        tenant_id=tenant_id,
                    )
                continue
            if matching_key is not None and self._relation_effect_spec(matching_key) == spec:
                continue
            if matching_key is not None:
                self.relation_store.delete_relation(
                    matching_key.source_uri,
                    matching_key.relation_type,
                    matching_key.target_uri,
                    tenant_id=tenant_id,
                )
            self.relation_store.add_relation(
                ContextRelation(
                    source_uri=str(spec["source_uri"]),
                    relation_type=str(spec["relation_type"]),
                    target_uri=str(spec["target_uri"]),
                    weight=float(spec.get("weight", 1.0)),
                    metadata=dict(spec.get("metadata", {}) or {}),
                ),
                tenant_id=tenant_id,
            )

    def _relation_effect_spec(self: OperationTransactionHost, relation: ContextRelation) -> dict:
        return {
            "source_uri": relation.source_uri,
            "relation_type": relation.relation_type,
            "target_uri": relation.target_uri,
            "weight": float(relation.weight),
            "metadata": dict(relation.metadata),
        }

    def _read_content_or_empty(self: OperationTransactionHost, uri: str) -> str:
        try:
            return self.source_store.read_content(uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return ""
