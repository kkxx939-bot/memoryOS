"""只负责保持跨领域启动恢复顺序，不承载领域恢复细节。"""

from __future__ import annotations

from typing import Any

from foundation.readiness import RuntimeReadinessState
from infrastructure.store.memory.owner_registry import bounded_owner_ids
from memory.commit.recovery import (
    recover_adoption_receipts,
    recover_memory_consolidations,
    recover_session_commit_groups,
)
from runtime.container import RuntimeContainer
from runtime.recovery.report import RecoveryReport

_MAX_STARTUP_MEMORY_OWNERS = 1_000


class RuntimeRecoveryCoordinator:
    """按显式顺序执行事务、记忆、会话和派生层恢复。"""

    def recover(self, runtime: RuntimeContainer) -> RecoveryReport:
        details: dict[str, Any] = {"runtime_layout": "markdown_memory_v1"}
        runtime.readiness.transition(RuntimeReadinessState.RECOVERING, details=details)
        try:
            details["queue_expired_leases"] = runtime.stores.queue.recover_expired_leases()
            details["ordinary_operations"] = runtime.transaction.recovery_worker.process_all()
            owners = bounded_owner_ids(
                runtime.layout,
                runtime.layout.tenant_id,
                _MAX_STARTUP_MEMORY_OWNERS,
            )
            details["owners"] = list(owners)
            for owner in owners:
                runtime.memory.document_store.probe_write_capabilities(
                    runtime.layout.tenant_id,
                    owner,
                )
            details["memory_source_filesystems_probed"] = len(owners)

            document_recovery: dict[str, Any] = {}
            for owner in owners:
                document_report = runtime.memory.committer.recover_all(runtime.layout.tenant_id, owner)
                if document_report.conflicted_intent_ids:
                    raise RuntimeError(
                        "document recovery preserved third-state external edits: "
                        + ",".join(document_report.conflicted_intent_ids)
                    )
                document_recovery[owner] = {"completed": len(document_report.completed), "conflicted": 0}
            details["document_intents"] = document_recovery

            erasure_recovery: dict[str, Any] = {}
            for owner in owners:
                erasure_report = runtime.memory.eraser.recover_owner(runtime.layout.tenant_id, owner)
                erasure_recovery[owner] = {
                    "completed": list(erasure_report.completed_document_ids),
                    "pending": list(erasure_report.pending_document_ids),
                }
            details["memory_document_erasures"] = erasure_recovery
            details["memory_document_adoptions"] = recover_adoption_receipts(
                runtime.memory,
                tenant_id=runtime.layout.tenant_id,
                owners=owners,
            )
            details["memory_consolidations_pre_projection"] = recover_memory_consolidations(
                runtime.memory.consolidator,
                tenant_id=runtime.layout.tenant_id,
                owners=owners,
            )
            details["session_commit_groups"] = recover_session_commit_groups(runtime.session.commit_service)
            details["session_archive_rebuild"] = runtime.session.commit_service.rebuild_session_archives()

            external_scan: dict[str, Any] = {}
            for owner in owners:
                result = runtime.memory.scanner.scan(
                    runtime.layout.tenant_id,
                    owner,
                    force_stable=True,
                )
                if result.deletions_paused:
                    raise RuntimeError(
                        f"memory scan deletion reconciliation paused for {owner}: {result.pause_reason}"
                    )
                external_scan[owner] = {
                    "confirmed": len(result.confirmed_changes),
                    "pending": result.pending_change_count,
                }
            details["memory_full_scan"] = external_scan

            rebuild: dict[str, Any] = {}
            for owner in owners:
                rebuild[owner] = runtime.memory.projection_worker.rebuild_owner(runtime.layout.tenant_id, owner)
            details["memory_document_rebuild"] = rebuild
            details["memory_projection_queue"] = runtime.memory.projection_worker.drain_until_quiescent()
            details["memory_consolidations_post_projection"] = recover_memory_consolidations(
                runtime.memory.consolidator,
                tenant_id=runtime.layout.tenant_id,
                owners=owners,
            )
            details["memory_projection_queue_after_consolidation"] = (
                runtime.memory.projection_worker.drain_until_quiescent()
            )
            details["generic_tombstones"] = runtime.context.tombstone_service.drain_pending(
                tenant_id=runtime.layout.tenant_id,
            )
            verified: dict[str, Any] = {}
            for owner in owners:
                verified[owner] = runtime.memory.projection_worker.verify_owner(runtime.layout.tenant_id, owner)
            details["memory_document_verification"] = verified
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
