from __future__ import annotations

from memoryos.contextdb.layers.layer_generator import l0_abstract, l1_overview
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.session.session_model import SessionArchive, SessionCommitResult
from memoryos.contextdb.store.source_store import QueueJob, QueueStore
from memoryos.core.ids import new_id


class SessionCommitService:
    def __init__(self, archive_store: SessionArchiveStore, queue_store: QueueStore) -> None:
        self.archive_store = archive_store
        self.queue_store = queue_store

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
        empty_diff = {"task_id": archive.task_id, "operations": [], "status": "pending_extraction"}
        self.archive_store.write_async_outputs(
            archive.archive_uri,
            abstract=abstract,
            overview=overview,
            memory_diff=empty_diff,
            behavior_diff=empty_diff,
            action_policy_diff=empty_diff,
            context_diff={"task_id": archive.task_id, "operations": [], "status": "layers_refreshed"},
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
