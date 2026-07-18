from __future__ import annotations

import json

from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore
from memoryos.contextdb.transaction.recovery import RecoveryService
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.commit.redo_log import RedoLog
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


def test_recovery_source_written_missing_source_is_quarantined_once(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    committer = OperationCommitter(source, index, str(tmp_path))
    redo = RedoLog(tmp_path)
    operation = ContextOperation(
        user_id="u1",
        context_type=ContextType.BEHAVIOR_CASE,
        action=OperationAction.REFRESH_LAYERS,
        target_uri="memoryos://user/u1/behavior_cases/missing",
        payload={"reason": "test"},
        operation_id="op_missing_source",
    )
    redo.begin(operation, phase="source_written")

    result = RecoveryService(redo, committer).recover("u1")

    assert result.recovered_count == 0
    assert not (tmp_path / "system" / "redo" / "op_missing_source.json").exists()
    assert result.quarantine_count == 1
    assert list((tmp_path / "system" / "quarantine" / "redo").glob("*.json"))
    audit_rows = [
        json.loads(line)
        for line in (tmp_path / "system" / "audit" / "u1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert audit_rows[-1]["event_type"] == "recovery_failed"
    assert audit_rows[-1]["payload"]["operation_id"] == "op_missing_source"
    assert audit_rows[-1]["payload"]["terminal"] == "quarantine"
    assert RecoveryService(redo, committer).recover("u1").failed_count == 0
