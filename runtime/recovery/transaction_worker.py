"""在运行时恢复阶段执行通用事务 RedoLog 恢复。"""

from __future__ import annotations

from typing import Any

from infrastructure.store.operation.redo import RedoControlFileError
from transaction.commit.recovery import RecoveryService


class RecoveryWorker:
    """汇总普通 Context 事务 RedoLog 的恢复结果并维护就绪状态。"""

    def __init__(self, recovery: RecoveryService) -> None:
        self.recovery = recovery

    def process_pending(self, user_id: str) -> dict:
        result = self.recovery.recover(user_id)
        payload = {
            "recovered_count": result.recovered_count,
            "operation_ids": result.operation_ids,
            "failed_count": result.failed_count,
            "quarantine_count": result.quarantine_count,
            "last_error": result.last_error,
        }
        self._fail_closed_if_incomplete(payload)
        return payload

    def process_all(self) -> dict:
        try:
            entries = self.recovery.redo_log.pending_entries()
        except RedoControlFileError as exc:
            payload = {
                "recovered_count": 0,
                "operation_ids": [],
                "failed_count": len(exc.records),
                "quarantine_count": len(exc.records),
                "last_error": type(exc).__name__,
            }
            self._fail_closed_if_incomplete(payload)
            return payload
        users = sorted({entry.user_id for entry in entries})
        totals: dict[str, Any] = {
            "recovered_count": 0,
            "operation_ids": [],
            "failed_count": 0,
            "quarantine_count": 0,
            "last_error": "",
        }
        for user_id in users:
            current = self.process_pending(user_id)
            totals["recovered_count"] += int(current["recovered_count"])
            totals["operation_ids"].extend(current["operation_ids"])
            totals["failed_count"] += int(current["failed_count"])
            totals["quarantine_count"] += int(current["quarantine_count"])
            if current["last_error"]:
                totals["last_error"] = current["last_error"]
        self._fail_closed_if_incomplete(totals)
        return totals

    def _fail_closed_if_incomplete(self, result: dict[str, Any]) -> None:
        failed = int(result.get("failed_count", 0) or 0)
        quarantined = int(result.get("quarantine_count", 0) or 0)
        if not failed and not quarantined:
            return
        readiness = getattr(self.recovery.committer.source_store, "readiness", None)
        mark_not_ready = getattr(readiness, "mark_not_ready", None)
        if not callable(mark_not_ready):
            return
        last_error = str(result.get("last_error") or "RecoveryIncomplete")
        mark_not_ready(
            "recovery left failed or quarantined authoritative artifacts: "
            f"failed={failed}, quarantine={quarantined}, error={last_error}",
            details={
                "artifact": "recovery",
                "failed_count": failed,
                "quarantine_count": quarantined,
                "last_error": last_error,
            },
        )
