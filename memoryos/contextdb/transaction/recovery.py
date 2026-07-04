from __future__ import annotations

from dataclasses import dataclass

from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.commit.redo_log import RedoLog
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.operation_status import OperationStatus


@dataclass(frozen=True)
class RecoveryResult:
    recovered_count: int
    operation_ids: list[str]


class RecoveryService:
    def __init__(self, redo_log: RedoLog, committer: OperationCommitter) -> None:
        self.redo_log = redo_log
        self.committer = committer

    def recover(self, user_id: str) -> RecoveryResult:
        entries = self.redo_log.pending_entries()
        if not entries:
            return RecoveryResult(recovered_count=0, operation_ids=[])
        recovered: list[str] = []
        for entry in entries:
            operation = entry.operation
            if entry.phase == "committed":
                self.redo_log.commit(operation.operation_id)
                continue
            if entry.phase in {"started", "begin"}:
                diff = self.committer.commit(user_id, [operation])
                recovered.extend(op.operation_id for op in diff.operations)
                continue
            if entry.phase == "source_written":
                self.committer._apply_index(operation)
                self.committer.redo.advance(operation, phase="index_written")
                self.committer.audit.record(user_id, "context_operation_committed", operation.to_dict())
                self.committer.redo.advance(operation, phase="audit_written")
                self._write_recovery_diff(user_id, operation)
                self.committer.redo.advance(operation, phase="diff_written")
                self.redo_log.commit(operation.operation_id)
                recovered.append(operation.operation_id)
                continue
            if entry.phase == "index_written":
                self.committer.audit.record(user_id, "context_operation_committed", operation.to_dict())
                self.committer.redo.advance(operation, phase="audit_written")
                self._write_recovery_diff(user_id, operation)
                self.committer.redo.advance(operation, phase="diff_written")
                self.redo_log.commit(operation.operation_id)
                recovered.append(operation.operation_id)
                continue
            if entry.phase == "audit_written":
                self._write_recovery_diff(user_id, operation)
                self.committer.redo.advance(operation, phase="diff_written")
                self.redo_log.commit(operation.operation_id)
                recovered.append(operation.operation_id)
                continue
            if entry.phase == "diff_written":
                self.redo_log.commit(operation.operation_id)
                recovered.append(operation.operation_id)
        return RecoveryResult(recovered_count=len(recovered), operation_ids=recovered)

    def _write_recovery_diff(self, user_id: str, operation) -> None:
        operation.status = OperationStatus.COMMITTED
        self.committer.diff_writer.write(ContextDiff(user_id=user_id, operations=[operation], diff_id=f"diff_{operation.operation_id}"))
