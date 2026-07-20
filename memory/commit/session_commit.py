"""Session 归档后的耐久提交协调入口。"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from behavior.execute.session_commit_planner import BehaviorCommitPlanner
from infrastructure.context.projection_journal import SessionProjectionJournal
from infrastructure.context.session_commit_planner import ContextCommitPlanner
from infrastructure.store.contracts.queue import QueueJob, QueueStore
from infrastructure.store.contracts.session_archive import SessionArchiveStore
from infrastructure.store.memory.layout import tenant_control_root
from infrastructure.store.session.commit_group import (
    CommitGroupStore,
)
from memory.commit.document_commit import (
    MemoryDocumentCommitter,
)
from memory.commit.entry import commit_session as commit_session_entry
from memory.commit.model.session import (
    SessionCommitResult,
    SessionCommitState,
)
from memory.commit.session_commit_types import (
    ConsumerLeaseBusy,
    ConsumerTerminalError,
    DerivedConsumerError,
)
from memory.commit.session_support import _SessionCommitSupport
from memory.execute.write_planner import MemoryDocumentPlanner
from policy.action_policy.planning.session_commit_planner import (
    ActionPolicyCommitPlanner,
)
from pre.session import SessionArchive
from transaction.commit.operation_committer import OperationCommitter


class SessionCommitService(_SessionCommitSupport):
    """先归档证据，再分别提交 Memory 与其他普通消费者。"""

    def __init__(
        self,
        archive_store: SessionArchiveStore,
        queue_store: QueueStore,
        committer: OperationCommitter | None = None,
        memory_planner: Any | None = None,
        behavior_planner: BehaviorCommitPlanner | None = None,
        action_policy_planner: ActionPolicyCommitPlanner | None = None,
        context_planner: ContextCommitPlanner | None = None,
        session_projector: Any | None = None,
        commit_group_store: CommitGroupStore | None = None,
        memory_committer: MemoryDocumentCommitter | None = None,
        document_planner: MemoryDocumentPlanner | None = None,
        projection_journal: SessionProjectionJournal | None = None,
    ) -> None:
        self.archive_store = archive_store
        self.queue_store = queue_store
        self.committer = committer
        self.memory_planner = memory_planner
        self.behavior_planner = behavior_planner or BehaviorCommitPlanner()
        self.action_policy_planner = action_policy_planner or ActionPolicyCommitPlanner()
        self.context_planner = context_planner or ContextCommitPlanner()
        self.session_projector = session_projector
        expected_control_root = tenant_control_root(archive_store.root, archive_store.tenant_id)
        if commit_group_store is not None and commit_group_store.artifact_root != expected_control_root.resolve(
            strict=False
        ):
            raise ValueError("CommitGroupStore root differs from the bound Session tenant")
        self.commit_group_store = commit_group_store or CommitGroupStore(expected_control_root)
        self.memory_committer = memory_committer
        planner_document = getattr(memory_planner, "document_planner", None)
        if document_planner is not None and planner_document is not None and planner_document is not document_planner:
            raise ValueError("Session and memory planners must share one document planner")
        self.document_planner = document_planner or (
            planner_document if isinstance(planner_document, MemoryDocumentPlanner) else None
        )
        journal_store = getattr(session_projector, "catalog_store", None)
        self.projection_journal = projection_journal or SessionProjectionJournal(journal_store)
        self._startup_recovery_group: ContextVar[str] = ContextVar(
            f"memoryos_session_startup_recovery_{id(self)}",
            default="",
        )

    def commit_session(
        self,
        archive: SessionArchive,
        *,
        async_commit: bool = True,
    ) -> SessionCommitResult:
        return commit_session_entry(self, archive, async_commit=async_commit)

    def sync_archive(
        self,
        archive: SessionArchive,
        *,
        enqueue_commit_job: bool = True,
    ) -> SessionCommitResult:
        """在投影或发布队列任务前，先耐久归档证据。"""

        self._require_runtime_ready()
        tenant_id = self._bind_archive_tenant(archive)
        tracking = False
        try:
            tracking = self._record_projection(archive, tenant_id=tenant_id, status="PENDING")
            self.archive_store.write_sync_archive(archive)
            projection, projection_status = self._project_session_archive(archive)
            if tracking:
                self._record_projection(archive, tenant_id=tenant_id, status="PROJECTED")
            if enqueue_commit_job:
                self._enqueue_session_commit(archive, tenant_id=tenant_id)
            return SessionCommitResult(
                task_id=archive.task_id,
                archive_uri=archive.archive_uri,
                status="queued",
                state=SessionCommitState.QUEUED,
                archive_committed=True,
                session_projection_status=projection_status,
                session_projected_count=int(getattr(projection, "projected", 0) or 0),
            )
        except Exception as exc:
            if tracking:
                self._record_projection(
                    archive,
                    tenant_id=tenant_id,
                    status="FAILED",
                    error=type(exc).__name__,
                )
            if self.archive_store.archive_exists(archive.archive_uri, tenant_id=tenant_id):
                self._enqueue_session_commit(archive, tenant_id=tenant_id)
            raise

    def enqueue_failed_inline_commit(self, archive: SessionArchive) -> QueueJob:
        """同步执行失败后，以同一个耐久任务身份重新入队。"""

        tenant_id = self._bind_archive_tenant(archive)
        if not self.archive_store.archive_exists(archive.archive_uri, tenant_id=tenant_id):
            raise RuntimeError("failed inline Session commit has no immutable archive")
        persisted = self.archive_store.read_archive(
            archive.archive_uri,
            tenant_id=tenant_id,
            manifest_digest=str(archive.manifest_digest or "") or None,
        )
        self._assert_archive_identity(archive, persisted)
        return self._enqueue_session_commit(persisted, tenant_id=tenant_id)

    def async_commit(self, archive: SessionArchive) -> SessionCommitResult:
        """让独立消费者提交一个经过精确校验的已归档 Session。"""

        self._require_runtime_ready()
        tenant_id = self._bind_archive_tenant(archive)
        tracking = False
        try:
            tracking = self._record_projection(archive, tenant_id=tenant_id, status="PENDING")
            requested_manifest = str(archive.manifest_digest or "")
            if not self.archive_store.archive_exists(archive.archive_uri, tenant_id=tenant_id):
                self.archive_store.write_sync_archive(archive)
                requested_manifest = archive.manifest_digest
            persisted = self.archive_store.read_archive(
                archive.archive_uri,
                tenant_id=tenant_id,
                manifest_digest=requested_manifest or None,
            )
            self._assert_archive_identity(archive, persisted, allow_materialized_digests=True)
            archive = persisted
            projection, projection_status = self._project_session_archive(archive)
            if tracking:
                self._record_projection(archive, tenant_id=tenant_id, status="PROJECTED")
        except Exception as exc:
            if tracking:
                self._record_projection(
                    archive,
                    tenant_id=tenant_id,
                    status="FAILED",
                    error=type(exc).__name__,
                )
            if self.archive_store.archive_exists(archive.archive_uri, tenant_id=tenant_id):
                self._enqueue_session_commit(archive, tenant_id=tenant_id)
            raise

        group_id = f"commit_group_{archive.task_id}"
        group = self.commit_group_store.create(
            group_id,
            task_id=archive.task_id,
            archive_uri=archive.archive_uri,
            user_id=archive.user_id,
            tenant_id=tenant_id,
            archive_digest=archive.archive_digest,
            manifest_digest=archive.manifest_digest,
        )
        if group.complete and self.archive_store.async_outputs_done_for_task(archive):
            return self._result(
                archive,
                group,
                projection_status=projection_status,
                projected_count=int(getattr(projection, "projected", 0) or 0),
            )

        failures: list[tuple[str, bool]] = []
        actions: tuple[tuple[str, Callable[[str], dict[str, Any]]], ...] = (
            ("memory", lambda attempt: self._commit_memory(archive, group_id, attempt)),
            (
                "behavior",
                self._skipped_action
                if self._is_coding_agent(archive)
                else lambda _attempt: self._commit_ordinary(
                    archive,
                    group_id,
                    "behavior",
                    self.behavior_planner.plan(archive),
                ),
            ),
            (
                "action_policy",
                self._skipped_action
                if self._is_coding_agent(archive)
                else lambda _attempt: self._commit_ordinary(
                    archive,
                    group_id,
                    "action_policy",
                    self.action_policy_planner.plan(archive),
                ),
            ),
            (
                "context",
                lambda _attempt: self._commit_ordinary(
                    archive,
                    group_id,
                    "context",
                    self.context_planner.plan(archive),
                ),
            ),
        )
        for consumer, action in actions:
            try:
                self._run_consumer(group_id, consumer, action)
            except Exception as exc:
                failures.append((consumer, self._is_retryable(exc)))

        refreshed = self.commit_group_store.load(group_id)
        if refreshed is None:
            raise KeyError(f"unknown commit group: {group_id}")
        group = refreshed
        if failures:
            self._write_outputs(archive, group, complete=False)
            raise DerivedConsumerError(failures)
        if not group.complete:
            raise DerivedConsumerError(
                tuple((name, item.retryable) for name, item in group.consumers.items() if item.status != "completed")
            )
        self._validate_persisted_memory_effects(group)
        self._write_outputs(archive, group, complete=True)
        if tracking:
            self._record_projection(archive, tenant_id=tenant_id, status="PENDING")
        try:
            projection, projection_status = self._project_session_archive(archive)
        except Exception as exc:
            if tracking:
                self._record_projection(
                    archive,
                    tenant_id=tenant_id,
                    status="FAILED",
                    error=type(exc).__name__,
                )
            raise
        if tracking:
            self._record_projection(archive, tenant_id=tenant_id, status="PROJECTED")
        return self._result(
            archive,
            group,
            projection_status=projection_status,
            projected_count=int(getattr(projection, "projected", 0) or 0),
        )


__all__ = [
    "ConsumerLeaseBusy",
    "ConsumerTerminalError",
    "DerivedConsumerError",
    "SessionCommitService",
]
