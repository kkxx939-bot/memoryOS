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
        pending = self.redo_log.pending()
        if not pending:
            return RecoveryResult(recovered_count=0, operation_ids=[])
        diff = self.committer.commit(user_id, pending)
        return RecoveryResult(
            recovered_count=len(diff.operations),
            operation_ids=[operation.operation_id for operation in diff.operations],
        )
