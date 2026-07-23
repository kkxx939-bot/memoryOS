"""Session 普通派生消费者执行。"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from typing import Any

from infrastructure.context.layers.generator import l0_abstract, l1_overview
from infrastructure.store.session.commit_group import (
    CONSUMERS,
    CommitGroupStatus,
)
from pre.session import SessionArchive
from runtime.session.commit_model import (
    SessionCommitResult,
    SessionCommitState,
)
from runtime.session.commit_recovery import _SessionCommitRecovery
from runtime.session.commit_types import (
    ConsumerLeaseBusy,
    ConsumerTerminalError,
)
from transaction.model.context_diff import ContextDiff
from transaction.model.context_operation import ContextOperation


class _SessionCommitConsumers(_SessionCommitRecovery):
    def _commit_ordinary(
        self,
        archive: SessionArchive,
        group_id: str,
        consumer: str,
        operations: Sequence[ContextOperation],
    ) -> dict[str, Any]:
        stabilized = self._stabilize_operations(archive, group_id, consumer, operations)
        if stabilized and self.committer is None:
            raise RuntimeError(f"{consumer} Session operations require OperationCommitter")
        diff = (
            self.committer.commit(archive.user_id, stabilized)
            if self.committer is not None
            else ContextDiff(
                user_id=archive.user_id,
                operations=[],
            )
        )
        operation_ids = [
            item.operation_id for item in (*diff.operations, *diff.pending_operations, *diff.rejected_operations)
        ]
        return {
            "status": "committed",
            "operation_count": len(operation_ids),
            "operation_ids": operation_ids,
            "diff_id": diff.diff_id,
            "skipped": False,
        }

    def _run_consumer(
        self,
        group_id: str,
        consumer: str,
        action: Callable[[str], dict[str, Any]],
    ) -> dict[str, Any]:
        if consumer not in CONSUMERS:
            raise ValueError(f"unsupported Session consumer: {consumer}")
        group = self.commit_group_store.load(group_id)
        if group is None:
            raise KeyError(f"unknown commit group: {group_id}")
        current = group.consumers[consumer]
        if current.status == "completed":
            return dict(current.summary)
        if current.status in {"dead_letter", "quarantine"} or (current.status == "failed" and not current.retryable):
            raise ConsumerTerminalError(f"{consumer} consumer is terminal")
        attempt_id = uuid.uuid4().hex
        if not self.commit_group_store.claim_consumer(
            group_id,
            consumer,
            attempt_id=attempt_id,
        ):
            raise ConsumerLeaseBusy(f"{consumer} consumer lease is unavailable")
        try:
            summary = action(attempt_id)
            self.commit_group_store.complete_consumer(
                group_id,
                consumer,
                attempt_id=attempt_id,
                summary=summary,
            )
            return summary
        except Exception as exc:
            self.commit_group_store.fail_consumer(
                group_id,
                consumer,
                type(exc).__name__,
                retryable=self._is_retryable(exc),
                attempt_id=attempt_id,
            )
            raise

    def _write_outputs(
        self,
        archive: SessionArchive,
        group: CommitGroupStatus,
        *,
        complete: bool,
    ) -> None:
        source_text = "\n".join(
            [
                *[str(item.get("content", item.get("text", ""))) for item in archive.messages],
                *[str(item.get("raw_text", item.get("scene", ""))) for item in archive.observations],
            ]
        )
        abstract = l0_abstract(source_text or f"Session {archive.session_id}")
        overview = l1_overview(
            f"Session {archive.session_id}",
            [
                f"messages: {len(archive.messages)}",
                f"observations: {len(archive.observations)}",
                f"predictions: {len(archive.predictions)}",
                f"feedback: {len(archive.feedback)}",
            ],
        )
        self.archive_store.write_async_outputs(
            archive.archive_uri,
            abstract=abstract,
            overview=overview,
            behavior_diff=self._ordinary_output(archive, group, "behavior"),
            action_policy_diff=self._ordinary_output(archive, group, "action_policy"),
            context_diff=self._ordinary_output(archive, group, "context"),
            tenant_id=group.tenant_id,
            commit_group_status=group.to_dict(),
            complete=complete,
            task_id=archive.task_id,
            created_at=group.created_at,
        )

    @staticmethod
    def _ordinary_output(
        archive: SessionArchive,
        group: CommitGroupStatus,
        consumer: str,
    ) -> dict[str, Any]:
        item = group.consumers[consumer]
        summary = dict(item.summary)
        return {
            "task_id": archive.task_id,
            "commit_group_id": group.group_id,
            "status": summary.get("status", item.status),
            "operation_count": int(summary.get("operation_count", 0) or 0),
            "operation_ids": list(summary.get("operation_ids", []) or []),
            "diff_id": str(summary.get("diff_id", "") or ""),
            "skipped": bool(summary.get("skipped", False)),
        }

    def _result(
        self,
        archive: SessionArchive,
        group: CommitGroupStatus,
        *,
        projection_status: str = "not_configured",
        projected_count: int = 0,
    ) -> SessionCommitResult:
        if group.complete:
            status = "done"
            state = SessionCommitState.COMMITTED
            done = True
        elif group.terminal:
            status = "dead_letter"
            state = SessionCommitState.DEAD_LETTER
            done = False
        else:
            status = "retrying"
            state = SessionCommitState.FAILED_RETRYABLE
            done = False
        return SessionCommitResult(
            task_id=archive.task_id,
            archive_uri=archive.archive_uri,
            status=status,
            done=done,
            state=state,
            commit_group_id=group.group_id,
            commit_group_status=group.to_dict(),
            archive_committed=True,
            session_projection_status=projection_status,
            session_projected_count=projected_count,
        )
