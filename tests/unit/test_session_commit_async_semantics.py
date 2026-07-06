from __future__ import annotations

from typing import Any, cast

from memoryos.api.sdk.client import MemoryOSClient
from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.session import SessionArchive, SessionArchiveStore, SessionCommitService
from memoryos.contextdb.store.local_stores import (
    FileSystemSourceStore,
    InMemoryIndexStore,
    InMemoryQueueStore,
    InMemoryRelationStore,
)
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.prediction.model.prediction_request import PredictionRequest
from memoryos.workers.session_commit_worker import SessionCommitWorker


class RecordingCommitter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list]] = []

    def commit(self, user_id: str, operations: list) -> ContextDiff:
        self.calls.append((user_id, list(operations)))
        return ContextDiff(user_id=user_id, operations=list(operations))


def _archive(session_id: str = "s1") -> SessionArchive:
    return SessionArchive(
        user_id="u1",
        session_id=session_id,
        archive_uri=f"memoryos://user/u1/sessions/history/{session_id}",
        messages=[{"role": "user", "content": "remember hot room preference"}],
        observations=[{"episode_id": session_id, "scene_key": "hot_room", "raw_text": "hot room"}],
        predictions=[{"observation": {"scene_key": "hot_room"}, "decision": {"action": "turn_on_ac"}}],
        feedback=[
            {
                "episode_id": session_id,
                "scene_key": "hot_room",
                "action": "turn_on_ac",
                "policy_uri": "memoryos://user/u1/action_policies/hot_room/turn_on_ac",
                "reward": 1.0,
            }
        ],
        used_contexts=[{"uri": "memoryos://user/u1/memories/anchors/hot_room", "context_type": "memory"}],
    )


def _context_db(tmp_path, queue: InMemoryQueueStore, committer: RecordingCommitter) -> ContextDB:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()
    service = SessionCommitService(SessionArchiveStore(tmp_path), queue, committer=cast(Any, committer))
    return ContextDB(source, index, relations, queue_store=queue, session_commit_service=service, committer=cast(Any, committer))


def test_commit_session_async_true_does_not_leave_worker_session_commit_job(tmp_path) -> None:
    queue = InMemoryQueueStore()
    committer = RecordingCommitter()
    db = _context_db(tmp_path, queue, committer)

    result = db.commit_session(_archive(), async_commit=True)

    assert result.status == "done"
    assert result.done is True
    assert queue.lease("session_commit", 1) == []
    assert committer.calls
    committed_once = sum(len(operations) for _, operations in committer.calls)
    assert committed_once > 0
    assert sum(1 for job in queue.jobs.values() if job.queue_name == "session_commit") == 0
    call_count_after_commit = len(committer.calls)
    assert queue.lease("session_commit", 1) == []
    assert len(committer.calls) == call_count_after_commit


def test_commit_session_async_false_keeps_sync_archive_pending_job(tmp_path) -> None:
    queue = InMemoryQueueStore()
    committer = RecordingCommitter()
    db = _context_db(tmp_path, queue, committer)

    result = db.commit_session(_archive(), async_commit=False)

    assert result.status == "queued"
    assert result.done is False
    pending = queue.lease("session_commit", 1)
    assert [job.job_id for job in pending] == [result.task_id]
    assert committer.calls == []


def test_session_commit_worker_processes_real_pending_archive_once(tmp_path) -> None:
    queue = InMemoryQueueStore()
    committer = RecordingCommitter()
    service = SessionCommitService(SessionArchiveStore(tmp_path), queue, committer=cast(Any, committer))
    archive = _archive()
    queued = service.sync_archive(archive)

    pending = queue.lease("session_commit", 1)
    assert [job.job_id for job in pending] == [queued.task_id]
    result = SessionCommitWorker(service).process_archive(archive)

    assert result == {"task_id": archive.task_id, "status": "done", "done": True}
    assert committer.calls
    committed_once = sum(len(operations) for _, operations in committer.calls)
    assert committed_once > 0
    queue.ack(pending[0].job_id)
    assert queue.lease("session_commit", 1) == []


def test_process_observation_async_true_returns_commit_result_without_session_commit_job(tmp_path) -> None:
    queue = InMemoryQueueStore()
    client = MemoryOSClient(str(tmp_path), queue_store=queue)
    request = PredictionRequest(
        user_id="u1",
        episode_id="ep1",
        observation="hot room",
        available_actions=["turn_on_ac", "ask_user", "do_nothing"],
        request_id="req1",
    )

    result = client.process_observation(request, archive_session=True, async_commit=True)

    assert result.session_commit_result is not None
    assert result.session_commit_result.status == "done"
    assert result.session_commit_result.done is True
    assert result.archive_uri == "memoryos://user/u1/sessions/history/ep1"
    assert queue.lease("session_commit", 1) == []


def test_process_observation_async_false_returns_queued_commit_result_and_worker_job(tmp_path) -> None:
    queue = InMemoryQueueStore()
    client = MemoryOSClient(str(tmp_path), queue_store=queue)
    request = PredictionRequest(
        user_id="u1",
        episode_id="ep1",
        observation="hot room",
        available_actions=["turn_on_ac", "ask_user", "do_nothing"],
        request_id="req1",
    )

    result = client.process_observation(request, archive_session=True, async_commit=False)

    assert result.prediction_result is not None
    assert result.session_commit_result is not None
    assert result.session_commit_result.status == "queued"
    assert result.session_commit_result.done is False
    assert result.archive_uri == "memoryos://user/u1/sessions/history/ep1"
    pending = queue.lease("session_commit", 1)
    assert [job.job_id for job in pending] == [result.session_commit_result.task_id]
