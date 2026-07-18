"""Durable recovery for ordinary operation redo entries."""

from __future__ import annotations

from dataclasses import dataclass

from memoryos.contextdb.store.lock_store import LockLostError
from memoryos.core.durable_io.quarantine import quarantine_control_file
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.commit.redo_log import RedoControlFileError, RedoIntegrityError, RedoLog


@dataclass(frozen=True)
class RecoveryResult:
    recovered_count: int
    operation_ids: list[str]
    failed_count: int = 0
    quarantine_count: int = 0
    last_error: str = ""


class RecoveryService:
    def __init__(self, redo_log: RedoLog, committer: OperationCommitter) -> None:
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

    def recover_outboxes(self) -> RecoveryResult:
        """The greenfield operation plane has no transaction outbox."""

        return RecoveryResult(0, [])

    def _quarantine(self, operation_id: str, exc: BaseException) -> int:
        path = self.redo_log.redo_dir / f"{operation_id}.json"
        if not path.exists() and not path.is_symlink():
            return 0
        quarantine_control_file(
            self.committer.artifact_root,
            path,
            kind="redo",
            error=exc,
            identifiers={"operation_id": operation_id},
        )
        return 1

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
