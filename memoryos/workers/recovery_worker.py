"""负责故障恢复的后台任务。"""

from __future__ import annotations

from typing import Any

from memoryos.contextdb.transaction.recovery import RecoveryService
from memoryos.operations.commit.redo_log import RedoControlFileError


class RecoveryWorker:
    def __init__(self, recovery: RecoveryService) -> None:
        self.recovery = recovery

    def process_pending(self, user_id: str) -> dict:
        result = self.recovery.recover(user_id)
        return {
            "recovered_count": result.recovered_count,
            "operation_ids": result.operation_ids,
            "failed_count": result.failed_count,
            "quarantine_count": result.quarantine_count,
            "last_error": result.last_error,
        }

    def process_all(self) -> dict:
        orphaned = self.recovery.recover_outboxes()
        try:
            entries = self.recovery.redo_log.pending_entries()
        except RedoControlFileError as exc:
            return {
                "recovered_count": orphaned.recovered_count,
                "operation_ids": orphaned.operation_ids,
                "failed_count": orphaned.failed_count + len(exc.records),
                "quarantine_count": orphaned.quarantine_count + len(exc.records),
                "last_error": type(exc).__name__,
            }
        users = sorted({entry.user_id for entry in entries})
        totals: dict[str, Any] = {
            "recovered_count": orphaned.recovered_count,
            "operation_ids": list(orphaned.operation_ids),
            "failed_count": orphaned.failed_count,
            "quarantine_count": orphaned.quarantine_count,
            "last_error": orphaned.last_error,
        }
        for user_id in users:
            current = self.process_pending(user_id)
            totals["recovered_count"] += int(current["recovered_count"])
            totals["operation_ids"].extend(current["operation_ids"])
            totals["failed_count"] += int(current["failed_count"])
            totals["quarantine_count"] += int(current["quarantine_count"])
            if current["last_error"]:
                totals["last_error"] = current["last_error"]
        return totals
