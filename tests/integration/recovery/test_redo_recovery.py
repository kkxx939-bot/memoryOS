from __future__ import annotations

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore
from memoryos.contextdb.transaction.recovery import RecoveryService
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


def test_source_written_recovery_rebuilds_index(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    committer = OperationCommitter(source, index, str(tmp_path))
    obj = ContextObject(
        uri="memoryos://user/u1/behavior_cases/temperature",
        context_type=ContextType.BEHAVIOR_CASE,
        title="temperature",
        owner_user_id="u1",
    )
    operation = ContextOperation(
        user_id="u1",
        context_type=ContextType.BEHAVIOR_CASE,
        action=OperationAction.ADD,
        target_uri=obj.uri,
        payload={"context_object": obj.to_dict(), "content": "prefers 26 degree"},
    )
    committer._apply_source(operation)
    committer.redo.begin(
        operation,
        phase="source_written",
        source_effect=committer._capture_regular_source_effect(operation),
    )

    result = RecoveryService(committer.redo, committer).recover("u1")

    assert result.operation_ids == [operation.operation_id]
    assert index.search(
        "26",
        tenant_id="default",
        filters={"owner_user_id": "u1", "context_type": "behavior_case"},
    )
    assert not committer.redo.pending()
