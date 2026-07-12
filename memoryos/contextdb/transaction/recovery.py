"""事务故障恢复。"""

from __future__ import annotations

from dataclasses import dataclass

from memoryos.contextdb.store.source_store import LockLostError
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.commit.redo_log import RedoIntegrityError, RedoLog


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
        canonical_by_transaction: dict[str, list] = {}
        regular_entries = []
        for entry in entries:
            if entry.operation.user_id != user_id:
                continue
            if entry.operation.payload.get("canonical_memory") is True:
                transaction_id = str(entry.operation.payload.get("transaction_id", ""))
                canonical_by_transaction.setdefault(transaction_id, []).append(entry)
            else:
                regular_entries.append(entry)
        for transaction_entries in canonical_by_transaction.values():
            try:
                recovered.extend(self.committer.resume_canonical_batch(user_id, transaction_entries))
            except (
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                LockLostError,
                RedoIntegrityError,
            ) as exc:
                for entry in transaction_entries:
                    self._record_failure(user_id, entry, exc)
        for entry in regular_entries:
            operation = entry.operation
            try:
                if self.committer.resume(
                    user_id,
                    operation,
                    entry.phase,
                    source_effect=entry.source_effect,
                    relation_manifest=entry.relation_manifest,
                ):
                    recovered.append(operation.operation_id)
            except (
                FileNotFoundError,
                IsADirectoryError,
                NotADirectoryError,
                LockLostError,
                RedoIntegrityError,
            ) as exc:
                self._record_failure(user_id, entry, exc)
        return RecoveryResult(recovered_count=len(recovered), operation_ids=recovered)

    def _record_failure(self, user_id: str, entry, exc: Exception) -> None:  # noqa: ANN001
        operation = entry.operation
        self.committer.audit.record(
            user_id,
            "recovery_failed",
            {
                "operation_id": operation.operation_id,
                "target_uri": operation.target_uri,
                "redo_phase": entry.phase,
                "error": str(exc),
            },
        )
