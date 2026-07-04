from __future__ import annotations

import json

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.action_policy.model.reward_signal import PenaltySignal, RewardSignal
from memoryos.action_policy.update.action_policy_updater import ActionPolicyUpdater
from memoryos.contextdb.layers.layer_refresher import LayerRefresher
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.local_stores import InMemoryLockStore
from memoryos.contextdb.store.source_store import IndexStore, LockStore, RelationStore, SourceStore
from memoryos.contextdb.transaction.path_lock import PathLock
from memoryos.core.time import utc_now
from memoryos.operations.commit.audit_writer import AuditWriter
from memoryos.operations.commit.diff_writer import DiffWriter
from memoryos.operations.commit.operation_coalescer import OperationCoalescer
from memoryos.operations.commit.redo_log import RedoLog
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus
from memoryos.operations.resolver.conflict_resolver import ConflictResolver
from memoryos.operations.resolver.target_resolver import TargetResolver


class OperationCommitter:
    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        root: str,
        lock_store: LockStore | None = None,
        relation_store: RelationStore | None = None,
        target_resolver: TargetResolver | None = None,
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.relation_store = relation_store
        self.coalescer = OperationCoalescer()
        self.conflicts = ConflictResolver()
        self.target_resolver = target_resolver or TargetResolver(index_store)
        self.redo = RedoLog(root)
        self.diff_writer = DiffWriter(root)
        self.audit = AuditWriter(root)
        self.path_lock = PathLock(lock_store or InMemoryLockStore())
        self.action_policy_updater = ActionPolicyUpdater()

    def commit(self, user_id: str, operations: list[ContextOperation]) -> ContextDiff:
        resolved_operations: list[ContextOperation] = []
        pending: list[ContextOperation] = []
        for operation in operations:
            result = self.target_resolver.resolve(operation, user_id=user_id)
            if result.resolved:
                resolved_operations.append(result.operation)
            else:
                result.operation.status = OperationStatus.PENDING
                pending.append(result.operation)
        conflict_result = self.conflicts.resolve(self._coalesce_non_policy_operations(resolved_operations))
        for operation in conflict_result.rejected:
            operation.status = OperationStatus.REJECTED
        committed = []
        for operation in conflict_result.accepted:
            if operation.status == OperationStatus.PENDING:
                pending.append(operation)
                continue
            lock_key = operation.target_uri or f"{operation.user_id}:{operation.operation_id}"
            self.redo.begin(operation, phase="started")
            with self.path_lock.acquire(lock_key):
                self._apply_source(operation)
                self.redo.advance(operation, phase="source_written")
                self._apply_index(operation)
                self.redo.advance(operation, phase="index_written")
                self.audit.record(user_id, "context_operation_committed", operation.to_dict())
                self.redo.advance(operation, phase="audit_written")
                operation.status = OperationStatus.COMMITTED
            committed.append(operation)
        diff = ContextDiff(user_id=user_id, operations=committed, pending_operations=pending, rejected_operations=conflict_result.rejected)
        self.diff_writer.write(diff)
        for operation in committed:
            self.redo.advance(operation, phase="diff_written")
            self.redo.commit(operation.operation_id)
        return diff

    def resume(self, user_id: str, operation: ContextOperation, phase: str) -> bool:
        if phase in {"committed"}:
            self.redo.commit(operation.operation_id)
            return False
        if phase in {"started", "begin"}:
            diff = self.commit(user_id, [operation])
            return any(op.operation_id == operation.operation_id for op in diff.operations)
        if phase == "source_written":
            self._apply_index(operation)
            self.redo.advance(operation, phase="index_written")
            self.audit.record(user_id, "context_operation_committed", operation.to_dict())
            self.redo.advance(operation, phase="audit_written")
            self._write_recovery_diff(user_id, operation)
            self.redo.advance(operation, phase="diff_written")
            self.redo.commit(operation.operation_id)
            return True
        if phase == "index_written":
            self.audit.record(user_id, "context_operation_committed", operation.to_dict())
            self.redo.advance(operation, phase="audit_written")
            self._write_recovery_diff(user_id, operation)
            self.redo.advance(operation, phase="diff_written")
            self.redo.commit(operation.operation_id)
            return True
        if phase == "audit_written":
            self._write_recovery_diff(user_id, operation)
            self.redo.advance(operation, phase="diff_written")
            self.redo.commit(operation.operation_id)
            return True
        if phase == "diff_written":
            self.redo.commit(operation.operation_id)
            return True
        return False

    def _write_recovery_diff(self, user_id: str, operation: ContextOperation) -> None:
        operation.status = OperationStatus.COMMITTED
        self.diff_writer.write(ContextDiff(user_id=user_id, operations=[operation], diff_id=f"diff_{operation.operation_id}"))

    def _coalesce_non_policy_operations(self, operations: list[ContextOperation]) -> list[ContextOperation]:
        policy_actions = {
            OperationAction.REWARD,
            OperationAction.PENALIZE,
            OperationAction.COOLDOWN,
            OperationAction.SUPPRESS,
            OperationAction.DISABLE,
        }
        policy_ops = [operation for operation in operations if operation.action in policy_actions]
        other_ops = [operation for operation in operations if operation.action not in policy_actions]
        return [*self.coalescer.coalesce(other_ops), *policy_ops]

    def _apply_source(self, operation: ContextOperation) -> None:
        if operation.action == OperationAction.SUPERSEDE:
            self._apply_supersede_source(operation)
            return
        if operation.action in {OperationAction.ADD, OperationAction.UPDATE}:
            object_payload = operation.payload.get("context_object")
            if isinstance(object_payload, dict):
                obj = ContextObject.from_dict(object_payload)
                content = str(operation.payload.get("content", ""))
                self.source_store.write_object(obj, content=content)
                self._apply_relations(obj, operation)
            return
        if operation.action in {
            OperationAction.REWARD,
            OperationAction.PENALIZE,
            OperationAction.COOLDOWN,
            OperationAction.SUPPRESS,
            OperationAction.DISABLE,
        } and operation.target_uri:
            if operation.context_type == ContextType.ACTION_POLICY:
                policy = self._read_action_policy(operation.target_uri)
                policy = self._apply_action_policy_mutation(policy, operation)
                self._write_action_policy(policy)
            elif operation.action == OperationAction.DISABLE:
                self.source_store.soft_delete(operation.target_uri, operation.action.value)
            return
        if operation.action == OperationAction.COMPRESS and operation.target_uri:
            obj = self.source_store.read_object(operation.target_uri)
            content = self._read_content_or_empty(operation.target_uri)
            LayerRefresher(self.source_store).refresh(obj, content, bullets=[operation.payload.get("reason", "compressed")])
            obj.lifecycle_state = LifecycleState.COLD
            obj.metadata = {**obj.metadata, "compressed_at": utc_now(), "compression_reason": operation.payload.get("reason", "")}
            self.source_store.write_object(obj)
            return
        if operation.action == OperationAction.REFRESH_LAYERS and operation.target_uri:
            obj = self.source_store.read_object(operation.target_uri)
            content = self._read_content_or_empty(operation.target_uri)
            LayerRefresher(self.source_store).refresh(obj, content)
            return
        if operation.action == OperationAction.ARCHIVE and operation.target_uri:
            obj = self.source_store.read_object(operation.target_uri)
            obj.lifecycle_state = LifecycleState.ARCHIVED
            obj.metadata = {**obj.metadata, "archived_at": utc_now(), "archive_reason": operation.payload.get("reason", "")}
            content = self._read_content_or_empty(operation.target_uri)
            self.source_store.write_object(obj, content=content)
            return
        if operation.action == OperationAction.DELETE and operation.target_uri:
            self.source_store.soft_delete(operation.target_uri, operation.action.value)
            return

    def _apply_index(self, operation: ContextOperation) -> None:
        if operation.action == OperationAction.SUPERSEDE:
            self._apply_supersede_index(operation)
            return
        if operation.action in {OperationAction.ADD, OperationAction.UPDATE}:
            object_payload = operation.payload.get("context_object")
            if isinstance(object_payload, dict):
                obj = ContextObject.from_dict(object_payload)
                self.index_store.upsert_index(obj, content=str(operation.payload.get("content", "")))
            return
        if operation.action == OperationAction.DELETE and operation.target_uri:
            self.index_store.delete_index(operation.target_uri)
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
                self.index_store.delete_index(operation.target_uri)
                return
            obj = self.source_store.read_object(operation.target_uri)
            self.index_store.upsert_index(obj, content=self._read_content_or_empty(operation.target_uri))

    def _apply_action_policy_mutation(self, policy: ActionPolicy, operation: ContextOperation) -> ActionPolicy:
        if operation.action == OperationAction.REWARD:
            return self.action_policy_updater.reward(policy, RewardSignal.from_payload(operation.payload), operation_id=operation.operation_id)
        if operation.action == OperationAction.PENALIZE:
            return self.action_policy_updater.penalize(policy, PenaltySignal.from_payload(operation.payload), operation_id=operation.operation_id)
        if operation.action == OperationAction.COOLDOWN:
            return self.action_policy_updater.cooldown(policy, operation.payload.get("cooldown_until"), operation_id=operation.operation_id)
        if operation.action == OperationAction.SUPPRESS:
            return self.action_policy_updater.suppress(policy, operation_id=operation.operation_id)
        if operation.action == OperationAction.DISABLE:
            return self.action_policy_updater.disable_auto_execute(policy, operation_id=operation.operation_id)
        return policy

    def _apply_supersede_source(self, operation: ContextOperation) -> None:
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

    def _apply_supersede_index(self, operation: ContextOperation) -> None:
        if not operation.target_uri:
            return
        old_obj = self.source_store.read_object(operation.target_uri)
        self.index_store.upsert_index(old_obj, content=self._read_content_or_empty(operation.target_uri))
        object_payload = operation.payload.get("context_object")
        if isinstance(object_payload, dict):
            new_uri = object_payload.get("uri")
            if not new_uri:
                return
            new_obj = self.source_store.read_object(str(new_uri))
            self.index_store.upsert_index(new_obj, content=str(operation.payload.get("content", "")))

    def _add_supersede_relations(self, old_obj: ContextObject, new_obj: ContextObject) -> None:
        metadata = {
            "tenant_id": new_obj.tenant_id or old_obj.tenant_id or "default",
            "owner_user_id": new_obj.owner_user_id or old_obj.owner_user_id,
        }
        self._add_relation(new_obj.uri, "supersedes", old_obj.uri, metadata)
        self._add_relation(old_obj.uri, "superseded_by", new_obj.uri, metadata)

    def _read_action_policy(self, uri: str) -> ActionPolicy:
        obj = self.source_store.read_object(uri)
        data = dict(obj.metadata)
        if not data:
            content = self._read_content_or_empty(uri)
            data = json.loads(content) if content else {}
        return ActionPolicy(**data)

    def _write_action_policy(self, policy: ActionPolicy) -> None:
        obj = policy.to_context_object()
        self.source_store.write_object(
            obj,
            content=json.dumps(policy.to_dict(), ensure_ascii=False, indent=2),
        )
        self._apply_relations(obj, ContextOperation(user_id=policy.user_id, context_type=ContextType.ACTION_POLICY, action=OperationAction.UPDATE, target_uri=policy.uri, payload={}))

    def _apply_relations(self, obj: ContextObject, operation: ContextOperation) -> None:
        if self.relation_store is None:
            return
        metadata = dict(obj.metadata)
        relation_metadata = {"tenant_id": obj.tenant_id or "default", "owner_user_id": obj.owner_user_id}
        if obj.context_type == ContextType.ACTION_POLICY:
            self._add_relation(obj.uri, "anchored_by", str(metadata.get("memory_anchor_uri", "")), relation_metadata)
            for uri in metadata.get("required_resource_uris", []) or []:
                self._add_relation(obj.uri, "requires_resource", str(uri), relation_metadata)
            for uri in metadata.get("required_skill_uris", []) or []:
                self._add_relation(obj.uri, "requires_skill", str(uri), relation_metadata)
            for uri in metadata.get("supported_behavior_pattern_uris", []) or []:
                self._add_relation(obj.uri, "supported_by", str(uri), relation_metadata)
            for uri in metadata.get("constrained_by_memory_uris", []) or []:
                self._add_relation(obj.uri, "constrained_by", str(uri), relation_metadata)
        elif obj.context_type in {ContextType.BEHAVIOR_PATTERN, ContextType.BEHAVIOR_CLUSTER}:
            self._add_relation(obj.uri, "anchored_by", str(metadata.get("memory_anchor_uri", "")), relation_metadata)
            for uri in metadata.get("case_refs", []) or []:
                self._add_relation(obj.uri, "aggregated_from", str(uri), relation_metadata)
            for uri in metadata.get("related_policy_uris", []) or metadata.get("policy_uris", []) or []:
                self._add_relation(str(uri), "supported_by", obj.uri, relation_metadata)
        elif obj.context_type == ContextType.MEMORY:
            for policy_uri in metadata.get("constrains_policy_uris", []) or []:
                self._add_relation(str(policy_uri), "constrained_by", obj.uri, relation_metadata)
            for behavior_uri in metadata.get("supporting_behavior_uris", []) or []:
                self._add_relation(obj.uri, "evidence_for", str(behavior_uri), relation_metadata)
        for relation in obj.relations:
            if self.relation_store is not None:
                self.relation_store.add_relation(relation)

    def _add_relation(self, source_uri: str, relation_type: str, target_uri: str, metadata: dict) -> None:
        if self.relation_store is None or not target_uri:
            return
        self.relation_store.add_relation(
            ContextRelation(
                source_uri=source_uri,
                relation_type=relation_type,
                target_uri=target_uri,
                metadata={key: value for key, value in metadata.items() if value is not None},
            )
        )

    def _read_content_or_empty(self, uri: str) -> str:
        try:
            return self.source_store.read_content(uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return ""
