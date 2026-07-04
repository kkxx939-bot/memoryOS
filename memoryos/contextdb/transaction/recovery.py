from __future__ import annotations

from dataclasses import dataclass

from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.commit.redo_log import RedoLog


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
            if self.committer.resume(user_id, operation, entry.phase):
                recovered.append(operation.operation_id)
        return RecoveryResult(recovered_count=len(recovered), operation_ids=recovered)
