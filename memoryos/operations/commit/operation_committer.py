from __future__ import annotations

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.store.source_store import IndexStore, SourceStore
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
    def __init__(self, source_store: SourceStore, index_store: IndexStore, root: str) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.coalescer = OperationCoalescer()
        self.conflicts = ConflictResolver()
        self.redo = RedoLog(root)
        self.diff_writer = DiffWriter(root)
        self.audit = AuditWriter(root)

    def commit(self, user_id: str, operations: list[ContextOperation]) -> ContextDiff:
        conflict_result = self.conflicts.resolve(self.coalescer.coalesce(operations))
        committed = []
        for operation in conflict_result.accepted:
            self.redo.begin(operation)
            self._apply(operation)
            operation.status = OperationStatus.COMMITTED
            self.redo.commit(operation.operation_id)
            committed.append(operation)
            self.audit.record(user_id, "context_operation_committed", operation.to_dict())
        diff = ContextDiff(user_id=user_id, operations=committed)
        self.diff_writer.write(diff)
        return diff

    def _apply(self, operation: ContextOperation) -> None:
        if operation.action in {OperationAction.ADD, OperationAction.UPDATE, OperationAction.SUPERSEDE}:
            object_payload = operation.payload.get("context_object")
            if isinstance(object_payload, dict):
                obj = ContextObject.from_dict(object_payload)
                content = str(operation.payload.get("content", ""))
                self.source_store.write_object(obj, content=content)
                self.index_store.upsert_index(obj, content=content)
            return
        if operation.action in {OperationAction.DELETE, OperationAction.DISABLE} and operation.target_uri:
            self.source_store.soft_delete(operation.target_uri, operation.action.value)
            self.index_store.delete_index(operation.target_uri)
            return
        if operation.action == OperationAction.REINDEX and operation.target_uri:
            obj = self.source_store.read_object(operation.target_uri)
            self.index_store.upsert_index(obj)
