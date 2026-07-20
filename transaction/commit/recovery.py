"""根据短期 Redo 意图恢复未完成的普通操作事务。"""

from __future__ import annotations

from dataclasses import dataclass

from infrastructure.store.contracts.lock import LockLostError
from transaction.commit.control import RedoControlFileError, RedoIntegrityError, RedoStore
from transaction.commit.operation_committer import OperationCommitter


@dataclass(frozen=True)
class RecoveryResult:
    recovered_count: int
    operation_ids: list[str]
    failed_count: int = 0
    quarantine_count: int = 0
    last_error: str = ""


class RecoveryService:
    def __init__(self, redo_log: RedoStore, committer: OperationCommitter) -> None:
        self.redo_log = redo_log
        self.committer = committer

    def recover(self, user_id: str) -> RecoveryResult:
        try:
            entries = self.redo_log.pending_entries()
        except RedoControlFileError as exc:
            return RecoveryResult(
                0,
                [],
                failed_count=len(exc.records),
                quarantine_count=len(exc.records),
                last_error=type(exc).__name__,
            )
        recovered: list[str] = []
        failed = quarantined = 0
        last_error = ""
        for entry in entries:
            if entry.user_id != user_id:
                continue
            try:
                if self.committer.resume(
                    user_id,
                    entry.operation,
                    entry.phase,
                    source_effect=entry.source_effect,
                    relation_manifest=entry.relation_manifest,
                ):
                    recovered.append(entry.operation_id)
            except (RedoIntegrityError, PermissionError, ValueError) as exc:
                failed += 1
                quarantined += self._quarantine(entry.operation_id, exc)
                last_error = self._describe_failure(exc)
                self._record_failure(user_id, entry.operation_id, exc, terminal="quarantine")
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError, LockLostError, OSError) as exc:
                failed += 1
                last_error = self._describe_failure(exc)
                self._record_failure(user_id, entry.operation_id, exc, terminal="retryable")
        return RecoveryResult(
            len(recovered),
            recovered,
            failed_count=failed,
            quarantine_count=quarantined,
            last_error=last_error,
        )

    def _quarantine(self, operation_id: str, exc: BaseException) -> int:
        return int(self.redo_log.quarantine(operation_id, exc))

    def _record_failure(
        self,
        user_id: str,
        operation_id: str,
        exc: BaseException,
        *,
        terminal: str,
    ) -> None:
        self.committer.audit.record(
            user_id,
            "recovery_failed",
            {
                "operation_id": operation_id,
                "error_type": type(exc).__name__,
                "terminal": terminal,
            },
        )

    @staticmethod
    def _describe_failure(exc: BaseException) -> str:
        message = " ".join(str(exc).split())[:500]
        return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


__all__ = ["RecoveryResult", "RecoveryService"]
