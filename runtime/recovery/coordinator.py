"""只负责保持跨领域启动恢复顺序，不承载领域恢复细节。"""

from __future__ import annotations

from typing import Any

from foundation.readiness import RuntimeReadinessState
from runtime.container import RuntimeContainer
from runtime.recovery.report import RecoveryReport
from runtime.recovery.session_commit import recover_session_commit_groups


class RuntimeRecoveryCoordinator:
    """按显式顺序执行事务、会话和派生层恢复。"""

    def recover(self, runtime: RuntimeContainer) -> RecoveryReport:
        details: dict[str, Any] = {"runtime_layout": "context_runtime_v1"}
        runtime.readiness.transition(RuntimeReadinessState.RECOVERING, details=details)
        try:
            details["queue_expired_leases"] = runtime.stores.queue.recover_expired_leases()
            details["ordinary_operations"] = runtime.transaction.recovery_worker.process_all()
            details["session_commit_groups"] = recover_session_commit_groups(runtime.session.commit_service)
            details["session_archive_rebuild"] = runtime.session.commit_service.rebuild_session_archives()
            details["generic_tombstones"] = runtime.context.tombstone_service.drain_pending(
                tenant_id=runtime.layout.tenant_id,
            )
        except Exception as exc:  # 启动恢复是可观测的失败关闭边界。
            reasons = (f"{type(exc).__name__}: {exc}",)
            runtime.readiness.transition(
                RuntimeReadinessState.NOT_READY,
                reasons=reasons,
                details=details,
            )
            return RecoveryReport(ready=False, details=details, reasons=reasons)
        runtime.readiness.transition(RuntimeReadinessState.READY, details=details)
        return RecoveryReport(ready=True, details=details)


__all__ = ["RuntimeRecoveryCoordinator"]
