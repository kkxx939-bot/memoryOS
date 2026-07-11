"""负责故障恢复的后台任务。"""

from __future__ import annotations

from memoryos.contextdb.transaction.recovery import RecoveryService


class RecoveryWorker:
    def __init__(self, recovery: RecoveryService) -> None:
        self.recovery = recovery

    def process_pending(self, user_id: str) -> dict:
        result = self.recovery.recover(user_id)
        return {"recovered_count": result.recovered_count, "operation_ids": result.operation_ids}
