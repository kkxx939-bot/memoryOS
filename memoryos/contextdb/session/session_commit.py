"""上下文数据库里的会话提交。"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from memoryos.contextdb.layers.layer_generator import l0_abstract, l1_overview
from memoryos.contextdb.session.commit_group import CommitGroupStatus, CommitGroupStore
from memoryos.contextdb.session.planners import (
    ActionPolicyCommitPlanner,
    BehaviorCommitPlanner,
    ContextCommitPlanner,
    MemoryCommitPlanner,
)
from memoryos.contextdb.session.planners.memory_commit_planner import MemoryExtractionBackendError
from memoryos.contextdb.session.planning import PlanningContext
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.session.session_model import SessionArchive, SessionCommitResult, SessionCommitState
from memoryos.contextdb.store.source_store import QueueJob, QueueStore
from memoryos.core.ids import stable_hash
from memoryos.memory.canonical.transaction import RevisionConflictError
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation


class DerivedConsumerError(RuntimeError):
    def __init__(self, consumer: str, failures: list[str]) -> None:
        self.consumer = consumer
        self.failures = tuple(failures)
        super().__init__(f"{consumer} consumer failed: {','.join(failures)}")


class SessionCommitService:
    """先归档会话，再规划并提交各类上下文变更。"""

    def __init__(
        self,
        archive_store: SessionArchiveStore,
        queue_store: QueueStore,
        committer: OperationCommitter | None = None,
        memory_planner: MemoryCommitPlanner | None = None,
        behavior_planner: BehaviorCommitPlanner | None = None,
        action_policy_planner: ActionPolicyCommitPlanner | None = None,
        context_planner: ContextCommitPlanner | None = None,
        allow_plan_only: bool = False,
        projection_worker=None,
        commit_group_store: CommitGroupStore | None = None,
    ) -> None:
        self.archive_store = archive_store
        self.queue_store = queue_store
        self.committer = committer
        self.memory_planner = memory_planner or MemoryCommitPlanner()
        self.behavior_planner = behavior_planner or BehaviorCommitPlanner()
        self.action_policy_planner = action_policy_planner or ActionPolicyCommitPlanner()
        self.context_planner = context_planner or ContextCommitPlanner()
        self.allow_plan_only = allow_plan_only
        self.projection_worker = projection_worker
        self.commit_group_store = commit_group_store or CommitGroupStore(archive_store.root)

    def sync_archive(self, archive: SessionArchive, *, enqueue_commit_job: bool = True) -> SessionCommitResult:
        """先把原始会话证据写稳，再投递异步提交任务。"""

        self.archive_store.write_sync_archive(archive)
        if enqueue_commit_job:
            self.queue_store.enqueue(
                QueueJob(
                    job_id=archive.task_id,
                    queue_name="session_commit",
                    action="async_session_commit",
                    target_uri=archive.archive_uri,
                    payload={
                        "user_id": archive.user_id,
                        "session_id": archive.session_id,
                        "tenant_id": self._tenant_id(archive),
                        "archive_digest": str(getattr(archive, "archive_digest", "") or ""),
                        "manifest_digest": str(getattr(archive, "manifest_digest", "") or ""),
                    },
                )
            )
        return SessionCommitResult(
            task_id=archive.task_id, archive_uri=archive.archive_uri, status="queued", state=SessionCommitState.QUEUED
        )

    def async_commit(self, archive: SessionArchive) -> SessionCommitResult:
        """根据已归档会话生成并提交记忆、行为和上下文变更。"""

        tenant_id = self._tenant_id(archive)
        requested_manifest = str(archive.manifest_digest or "")
        if not self.archive_store.archive_exists(archive.archive_uri, tenant_id=tenant_id):
            self.archive_store.write_sync_archive(archive)
            requested_manifest = archive.manifest_digest
        archive = self.archive_store.read_archive(
            archive.archive_uri,
            tenant_id=tenant_id,
            manifest_digest=requested_manifest or None,
        )
        tenant_id = self._tenant_id(archive)
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
        if self.committer is None and not self.allow_plan_only:
            raise RuntimeError("SessionCommitService requires OperationCommitter unless allow_plan_only=True")
        if group.complete and self.archive_store.async_outputs_done_for_task(archive):
            return self._result(archive, group)
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
                "Long-term memory, behavior, action policy, and context diffs are emitted separately.",
            ],
        )
        if group.canonical_status != "completed":
            if group.canonical_status == "failed" and not group.canonical_retryable:
                return self._write_incomplete_outputs(
                    archive,
                    group,
                    abstract,
                    overview,
                    memory_diff={
                        "status": "failed",
                        "error": group.canonical_last_error,
                        "operation_count": 0,
                        "operations": [],
                    },
                )
            canonical_attempt_id = uuid.uuid4().hex
            if not self.commit_group_store.claim_canonical(
                group_id,
                attempt_id=canonical_attempt_id,
            ):
                current = self.commit_group_store.load(group_id)
                assert current is not None
                return self._result(archive, current)
            try:
                memory_result = self.memory_planner.plan(archive)
                memory_ops = list(memory_result.operations)
                memory_diff = self._commit_memory_with_reconcile_retry(
                    archive,
                    memory_ops,
                    memory_result.context,
                )
            except MemoryExtractionBackendError as exc:
                self._enqueue_memory_proposal(archive, exc)
                group = self.commit_group_store.fail_canonical(
                    group_id,
                    f"{type(exc).__name__}: {exc.error_type}",
                    retryable=exc.retryable,
                    attempt_id=canonical_attempt_id,
                )
                return self._write_incomplete_outputs(
                    archive,
                    group,
                    abstract,
                    overview,
                    memory_diff={
                        "status": "pending",
                        "proposal_status": "queued",
                        "error": exc.error_type,
                        "operation_count": 0,
                        "operations": [],
                    },
                )
            except (OSError, TimeoutError, RevisionConflictError) as exc:
                self.commit_group_store.fail_canonical(
                    group_id,
                    f"{type(exc).__name__}: {exc}",
                    retryable=True,
                    attempt_id=canonical_attempt_id,
                )
                raise
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                self.commit_group_store.fail_canonical(
                    group_id,
                    f"{type(exc).__name__}: {exc}",
                    retryable=False,
                    attempt_id=canonical_attempt_id,
                )
                raise
            group = self.commit_group_store.mark_canonical(
                group_id,
                revision=self._max_revision_from_diff(memory_diff),
                result=memory_diff,
                attempt_id=canonical_attempt_id,
            )
        memory_diff = dict(group.canonical_result)
        coding_agent = self._is_coding_agent(archive)
        projection_diff = self._run_consumer(
            group_id,
            "projection",
            lambda: self._project_commit_group(group_id, memory_diff),
        )
        memory_diff["projection"] = projection_diff
        if coding_agent:
            behavior_diff = self._complete_skipped_consumer(group_id, "behavior")
        else:
            behavior_diff = self._run_consumer(
                group_id,
                "behavior",
                lambda: self._commit_consumer_operations(
                    archive,
                    group_id,
                    "behavior",
                    self.behavior_planner.plan(archive),
                ),
            )
        if coding_agent:
            action_policy_diff = self._complete_skipped_consumer(group_id, "action_policy")
        else:
            action_policy_diff = self._run_consumer(
                group_id,
                "action_policy",
                lambda: self._commit_consumer_operations(
                    archive,
                    group_id,
                    "action_policy",
                    self.action_policy_planner.plan(archive),
                ),
            )
        context_diff = self._run_consumer(
            group_id,
            "context",
            lambda: self._commit_consumer_operations(
                archive,
                group_id,
                "context",
                self.context_planner.plan(archive),
            ),
        )
        refreshed_group = self.commit_group_store.load(group_id)
        assert refreshed_group is not None
        group = refreshed_group
        self.archive_store.write_async_outputs(
            archive.archive_uri,
            abstract=abstract,
            overview=overview,
            memory_diff={"task_id": archive.task_id, "commit_group_id": group_id, **memory_diff},
            behavior_diff={"task_id": archive.task_id, **behavior_diff},
            action_policy_diff={"task_id": archive.task_id, **action_policy_diff},
            context_diff={"task_id": archive.task_id, **context_diff},
            tenant_id=tenant_id,
            commit_group_status=group.to_dict(),
            complete=group.complete,
        )
        if group.complete:
            self._enqueue_refresh_consumers(archive, group_id)
        return self._result(archive, group)

    def _commit_memory_with_reconcile_retry(
        self,
        archive: SessionArchive,
        operations: list[ContextOperation],
        planning_context: PlanningContext | None = None,
    ) -> dict:
        try:
            return self._commit_or_describe(archive.user_id, operations)
        except RevisionConflictError as exc:
            if self.committer is not None:
                self.committer.recover_pending_canonical(archive.user_id)
            if planning_context is None:
                raise RevisionConflictError("revision conflict has no request-scoped PlanningContext") from exc
            replanned = self.memory_planner.replan(planning_context, archive)
            return self._commit_or_describe(archive.user_id, list(replanned.operations))

    def _run_consumer(
        self,
        group_id: str,
        consumer: str,
        action: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        status = self.commit_group_store.load(group_id)
        if status is None:
            raise KeyError(f"unknown commit group: {group_id}")
        existing = status.consumers[consumer]
        if existing.status == "completed":
            return dict(existing.result) or {"status": "completed"}
        if existing.status == "failed" and not existing.retryable:
            return {"status": "failed", "error": existing.last_error, "retryable": False}
        attempt_id = uuid.uuid4().hex
        if not self.commit_group_store.claim_consumer(
            group_id,
            consumer,
            attempt_id=attempt_id,
        ):
            current = self.commit_group_store.load(group_id)
            assert current is not None
            item = current.consumers[consumer]
            return {
                "status": item.status,
                "error": item.last_error,
                "retryable": item.retryable,
            }
        try:
            result = action()
            failures = [str(item) for item in result.get("failed", []) or []]
            if failures:
                raise DerivedConsumerError(consumer, failures)
        except (OSError, TimeoutError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            self.commit_group_store.fail_consumer(
                group_id,
                consumer,
                f"{type(exc).__name__}: {exc}",
                retryable=True,
                attempt_id=attempt_id,
            )
            return {"status": "failed", "error": type(exc).__name__, "retryable": True}
        current = self.commit_group_store.load(group_id)
        assert current is not None
        self.commit_group_store.complete_consumer(
            group_id,
            consumer,
            revision=current.canonical_revision,
            attempt_id=attempt_id,
            result=result,
        )
        return result

    def _complete_skipped_consumer(self, group_id: str, consumer: str) -> dict[str, Any]:
        result = {"status": "skipped", "operations": [], "operation_count": 0}
        status = self.commit_group_store.load(group_id)
        if status is None:
            raise KeyError(f"unknown commit group: {group_id}")
        if status.consumers[consumer].status != "completed":
            self.commit_group_store.complete_consumer(
                group_id,
                consumer,
                revision=status.canonical_revision,
                result=result,
            )
        return result

    def _project_commit_group(self, group_id: str, memory_diff: dict[str, Any]) -> dict[str, Any]:
        if self.projection_worker is None:
            return {"status": "skipped", "processed": [], "stale": [], "failed": []}
        transaction_ids = tuple(
            dict.fromkeys(
                str(operation.get("payload", {}).get("transaction_id", ""))
                for operation in memory_diff.get("operations", []) or []
                if isinstance(operation, dict) and operation.get("payload", {}).get("transaction_id")
            )
        )
        result = self.projection_worker.process_commit_group(
            group_id,
            transaction_ids=transaction_ids,
        )
        return {"status": "completed" if not result["failed"] else "failed", **result}

    def _commit_consumer_operations(
        self,
        archive: SessionArchive,
        group_id: str,
        consumer: str,
        operations: list[ContextOperation],
    ) -> dict[str, Any]:
        self._stabilize_consumer_operations(archive, group_id, consumer, operations)
        return self._commit_or_describe(archive.user_id, operations)

    def _stabilize_consumer_operations(
        self,
        archive: SessionArchive,
        group_id: str,
        consumer: str,
        operations: list[ContextOperation],
    ) -> None:
        for index, operation in enumerate(operations):
            operation.operation_id = f"op_{stable_hash([group_id, consumer, index, operation.action.value, operation.target_uri], length=32)}"
            operation.created_at = archive.created_at
            operation.payload["commit_group_id"] = group_id
            operation.payload["commit_consumer"] = consumer

    def _enqueue_memory_proposal(self, archive: SessionArchive, exc: MemoryExtractionBackendError) -> None:
        self.queue_store.enqueue(
            QueueJob(
                job_id=f"memory_proposal_{archive.task_id}",
                queue_name="memory_proposal",
                action="retry_memory_proposal",
                target_uri=archive.archive_uri,
                payload={
                    "task_id": archive.task_id,
                    "tenant_id": self._tenant_id(archive),
                    "archive_digest": archive.archive_digest,
                    "manifest_digest": archive.manifest_digest,
                    "error_type": exc.error_type,
                },
            )
        )

    def _enqueue_refresh_consumers(self, archive: SessionArchive, group_id: str) -> None:
        for queue_name in ("semantic", "embedding", "reindex"):
            self.queue_store.enqueue(
                QueueJob(
                    job_id=f"{queue_name}_{stable_hash([group_id, queue_name], length=32)}",
                    queue_name=queue_name,
                    action=f"{queue_name}_refresh",
                    target_uri=archive.archive_uri,
                    payload={"task_id": archive.task_id, "commit_group_id": group_id},
                )
            )

    def _write_incomplete_outputs(
        self,
        archive: SessionArchive,
        group: CommitGroupStatus,
        abstract: str,
        overview: str,
        *,
        memory_diff: dict[str, Any],
    ) -> SessionCommitResult:
        pending = {"status": "pending", "operations": [], "operation_count": 0}
        self.archive_store.write_async_outputs(
            archive.archive_uri,
            abstract=abstract,
            overview=overview,
            memory_diff={"task_id": archive.task_id, "commit_group_id": group.group_id, **memory_diff},
            behavior_diff={"task_id": archive.task_id, **pending},
            action_policy_diff={"task_id": archive.task_id, **pending},
            context_diff={"task_id": archive.task_id, **pending},
            tenant_id=group.tenant_id,
            commit_group_status=group.to_dict(),
            complete=False,
        )
        return self._result(archive, group)

    def _result(self, archive: SessionArchive, group: CommitGroupStatus) -> SessionCommitResult:
        failed_consumers = [name for name, item in group.consumers.items() if item.status == "failed"]
        if group.complete:
            status = "done"
        elif group.canonical_status == "completed" and failed_consumers:
            status = "derived_failed"
        elif group.canonical_status == "completed":
            status = "canonical_committed"
        else:
            status = "canonical_pending"
        return SessionCommitResult(
            task_id=archive.task_id,
            archive_uri=archive.archive_uri,
            status=status,
            done=group.complete,
            state=SessionCommitState.COMMITTED
            if group.canonical_status == "completed"
            else SessionCommitState.PROCESSING,
            commit_group_id=group.group_id,
            canonical_committed=group.canonical_status == "completed",
            commit_group_status=group.to_dict(),
        )

    def _tenant_id(self, archive: SessionArchive) -> str:
        metadata = dict(archive.metadata or {})
        return str(metadata.get("tenant_id") or dict(metadata.get("scope", {}) or {}).get("tenant_id") or "default")

    def _max_revision_from_diff(self, diff: dict[str, Any]) -> int | None:
        revisions = []
        for operation in diff.get("operations", []) or []:
            payload = operation.get("payload", {}).get("context_object") if isinstance(operation, dict) else None
            if isinstance(payload, dict):
                value = dict(payload.get("metadata", {}) or {}).get("revision")
                if value is not None:
                    revisions.append(int(value))
        return max(revisions) if revisions else None

    def _is_coding_agent(self, archive: SessionArchive) -> bool:
        connect = dict(archive.metadata.get("connect", {}) or {})
        return connect.get("connect_type") == "agent" and connect.get("run_mode") == "context_reduction"

    def _commit_or_describe(self, user_id: str, operations: list[ContextOperation]) -> dict:
        if self.committer is not None and operations:
            diff = self.committer.commit(user_id, operations)
            return self._diff_payload(diff, status="committed")
        if self.committer is not None:
            return {"status": "committed", "operations": [], "operation_count": 0}
        return {
            "status": "planned",
            "operations": [operation.to_dict() for operation in operations],
            "operation_count": len(operations),
        }

    def _diff_payload(self, diff: ContextDiff, status: str) -> dict:
        payload = diff.to_dict()
        payload["status"] = status
        payload["operation_count"] = len(diff.operations)
        return payload
