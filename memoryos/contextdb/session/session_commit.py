from __future__ import annotations

from memoryos.contextdb.layers.layer_generator import l0_abstract, l1_overview
from memoryos.contextdb.session.planners import (
    ActionPolicyCommitPlanner,
    BehaviorCommitPlanner,
    ContextCommitPlanner,
    MemoryCommitPlanner,
)
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.session.session_model import SessionArchive, SessionCommitResult, SessionCommitState
from memoryos.contextdb.store.source_store import QueueJob, QueueStore
from memoryos.core.ids import new_id
from memoryos.memory.canonical.transaction import RevisionConflictError
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation


class SessionCommitService:
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

    def sync_archive(self, archive: SessionArchive, *, enqueue_commit_job: bool = True) -> SessionCommitResult:
        self.archive_store.write_sync_archive(archive)
        if enqueue_commit_job:
            self.queue_store.enqueue(
                QueueJob(
                    job_id=archive.task_id,
                    queue_name="session_commit",
                    action="async_session_commit",
                    target_uri=archive.archive_uri,
                    payload={"user_id": archive.user_id, "session_id": archive.session_id},
                )
            )
        return SessionCommitResult(
            task_id=archive.task_id, archive_uri=archive.archive_uri, status="queued", state=SessionCommitState.QUEUED
        )

    def async_commit(self, archive: SessionArchive) -> SessionCommitResult:
        if self.archive_store.async_outputs_done_for_task(archive):
            return SessionCommitResult(
                task_id=archive.task_id,
                archive_uri=archive.archive_uri,
                status="done",
                done=True,
                state=SessionCommitState.COMMITTED,
            )
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
        memory_ops, memory_deferred = self._plan_memory(archive)
        coding_agent = self._is_coding_agent(archive)
        behavior_ops = [] if coding_agent else self.behavior_planner.plan(archive)
        if self.committer is None and not self.allow_plan_only:
            raise RuntimeError("SessionCommitService requires OperationCommitter unless allow_plan_only=True")
        memory_diff = self._commit_memory_with_reconcile_retry(archive, memory_ops)
        if memory_deferred:
            memory_diff["proposal_status"] = "queued"
        if self.projection_worker is not None:
            try:
                memory_diff["projection"] = self.projection_worker.process_pending()
            except Exception as exc:
                memory_diff["projection"] = {
                    "processed": [],
                    "failed": [type(exc).__name__],
                }
        behavior_diff = self._commit_or_describe(archive.user_id, behavior_ops)
        action_policy_ops = [] if coding_agent else self.action_policy_planner.plan(archive)
        action_policy_diff = self._commit_or_describe(archive.user_id, action_policy_ops)
        context_ops = self.context_planner.plan(archive)
        context_diff = self._commit_or_describe(archive.user_id, context_ops)
        self.archive_store.write_async_outputs(
            archive.archive_uri,
            abstract=abstract,
            overview=overview,
            memory_diff={"task_id": archive.task_id, **memory_diff},
            behavior_diff={"task_id": archive.task_id, **behavior_diff},
            action_policy_diff={"task_id": archive.task_id, **action_policy_diff},
            context_diff={"task_id": archive.task_id, **context_diff},
        )
        for queue_name in ("semantic", "embedding", "reindex"):
            self.queue_store.enqueue(
                QueueJob(
                    job_id=new_id(queue_name),
                    queue_name=queue_name,
                    action=f"{queue_name}_refresh",
                    target_uri=archive.archive_uri,
                    payload={"task_id": archive.task_id},
                )
            )
        return SessionCommitResult(
            task_id=archive.task_id,
            archive_uri=archive.archive_uri,
            status="done",
            done=True,
            state=SessionCommitState.COMMITTED,
        )

    def _plan_memory(self, archive: SessionArchive) -> tuple[list[ContextOperation], bool]:
        try:
            return self.memory_planner.plan(archive), False
        except Exception as exc:
            self.queue_store.enqueue(
                QueueJob(
                    job_id=f"memory_proposal_{archive.task_id}",
                    queue_name="memory_proposal",
                    action="retry_memory_proposal",
                    target_uri=archive.archive_uri,
                    payload={"task_id": archive.task_id, "error_type": type(exc).__name__},
                )
            )
            return [], True

    def _commit_memory_with_reconcile_retry(
        self,
        archive: SessionArchive,
        operations: list[ContextOperation],
    ) -> dict:
        try:
            return self._commit_or_describe(archive.user_id, operations)
        except RevisionConflictError:
            if self.committer is not None:
                self.committer.recover_pending_canonical(archive.user_id)
            replanned = self.memory_planner.replan_last(archive)
            return self._commit_or_describe(archive.user_id, replanned)

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
