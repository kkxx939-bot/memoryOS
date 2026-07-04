from __future__ import annotations

import json

from memoryos.action_policy.model.action_policy import ActionPolicy, ActionPolicyStatus
from memoryos.action_policy.model.reward_signal import PenaltySignal, RewardSignal
from memoryos.action_policy.update.action_policy_updater import ActionPolicyUpdater
from memoryos.contextdb.layers.layer_refresher import LayerRefresher
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.local_stores import InMemoryLockStore
from memoryos.contextdb.store.source_store import IndexStore, LockStore, SourceStore
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


class OperationCommitter:
    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        root: str,
        lock_store: LockStore | None = None,
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.coalescer = OperationCoalescer()
        self.conflicts = ConflictResolver()
        self.redo = RedoLog(root)
        self.diff_writer = DiffWriter(root)
        self.audit = AuditWriter(root)
        self.path_lock = PathLock(lock_store or InMemoryLockStore())
        self.action_policy_updater = ActionPolicyUpdater()

    def commit(self, user_id: str, operations: list[ContextOperation]) -> ContextDiff:
        conflict_result = self.conflicts.resolve(self.coalescer.coalesce(operations))
        committed = []
        for operation in conflict_result.accepted:
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
        diff = ContextDiff(user_id=user_id, operations=committed)
        self.diff_writer.write(diff)
        for operation in committed:
            self.redo.advance(operation, phase="diff_written")
            self.redo.commit(operation.operation_id)
        return diff

    def _apply_source(self, operation: ContextOperation) -> None:
        if operation.action in {OperationAction.ADD, OperationAction.UPDATE, OperationAction.SUPERSEDE}:
            object_payload = operation.payload.get("context_object")
            if isinstance(object_payload, dict):
                obj = ContextObject.from_dict(object_payload)
                content = str(operation.payload.get("content", ""))
                self.source_store.write_object(obj, content=content)
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
        if operation.action in {OperationAction.ADD, OperationAction.UPDATE, OperationAction.SUPERSEDE}:
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
            return self.action_policy_updater.reward(policy, RewardSignal.from_payload(operation.payload))
        if operation.action == OperationAction.PENALIZE:
            return self.action_policy_updater.penalize(policy, PenaltySignal.from_payload(operation.payload))
        if operation.action == OperationAction.COOLDOWN:
            policy.status = ActionPolicyStatus.COOLDOWN
            policy.cooldown_until = operation.payload.get("cooldown_until")
            policy.updated_at = utc_now()
            return policy
        if operation.action == OperationAction.SUPPRESS:
            return self.action_policy_updater.suppress(policy)
        if operation.action == OperationAction.DISABLE:
            return self.action_policy_updater.disable_auto_execute(policy)
        return policy

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

    def _read_content_or_empty(self, uri: str) -> str:
        try:
            return self.source_store.read_content(uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return ""
