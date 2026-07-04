from __future__ import annotations

from memoryos.contextdb.layers.layer_generator import l0_abstract, l1_overview
from memoryos.contextdb.session.planners import (
    ActionPolicyCommitPlanner,
    BehaviorCommitPlanner,
    ContextCommitPlanner,
    MemoryCommitPlanner,
)
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.session.session_model import SessionArchive, SessionCommitResult
from memoryos.contextdb.store.source_store import QueueJob, QueueStore
from memoryos.core.ids import new_id
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
    ) -> None:
        self.archive_store = archive_store
        self.queue_store = queue_store
        self.committer = committer
        self.memory_planner = memory_planner or MemoryCommitPlanner()
        self.behavior_planner = behavior_planner or BehaviorCommitPlanner()
        self.action_policy_planner = action_policy_planner or ActionPolicyCommitPlanner()
        self.context_planner = context_planner or ContextCommitPlanner()

    def sync_archive(self, archive: SessionArchive) -> SessionCommitResult:
        self.archive_store.write_sync_archive(archive)
        self.queue_store.enqueue(
            QueueJob(
                job_id=archive.task_id,
                queue_name="session_commit",
                action="async_session_commit",
                target_uri=archive.archive_uri,
                payload={"user_id": archive.user_id, "session_id": archive.session_id},
            )
        )
        return SessionCommitResult(task_id=archive.task_id, archive_uri=archive.archive_uri, status="queued")

    def async_commit(self, archive: SessionArchive) -> SessionCommitResult:
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
        memory_ops = self.memory_planner.plan(archive)
        behavior_ops = self.behavior_planner.plan(archive)
        action_policy_ops = self.action_policy_planner.plan(archive)
        context_ops = self.context_planner.plan(archive)
        memory_diff = self._commit_or_describe(archive.user_id, memory_ops)
        behavior_diff = self._commit_or_describe(archive.user_id, behavior_ops)
        action_policy_diff = self._commit_or_describe(archive.user_id, action_policy_ops)
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
        return SessionCommitResult(task_id=archive.task_id, archive_uri=archive.archive_uri, status="done", done=True)

    def _commit_or_describe(self, user_id: str, operations: list[ContextOperation]) -> dict:
        if self.committer is not None and operations:
            diff = self.committer.commit(user_id, operations)
            return self._diff_payload(diff, status="committed")
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
