from __future__ import annotations

from typing import Any, cast

from memoryos.api.mcp.config import MCPServerConfig
from memoryos.api.mcp.server import MemoryOSMCPServer
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.connect import ConnectMetadata
from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.session import SessionArchive, SessionArchiveStore, SessionCommitService
from memoryos.contextdb.session.planners.memory_commit_planner import MemoryCommitPlanner
from memoryos.contextdb.store.local_stores import (
    FileSystemSourceStore,
    InMemoryIndexStore,
    InMemoryQueueStore,
    InMemoryRelationStore,
)
from memoryos.memory.extraction import RuleFallbackExtractor
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.prediction.model.prediction_request import PredictionRequest
from memoryos.workers.session_commit_worker import SessionCommitWorker


class RecordingCommitter:
    def __init__(self, delegate: OperationCommitter) -> None:
        self.delegate = delegate
        self.source_store = delegate.source_store
        self.calls: list[tuple[str, list[ContextOperation]]] = []

    def commit(self, user_id: str, operations: list[ContextOperation]) -> ContextDiff:
        self.calls.append((user_id, list(operations)))
        return self.delegate.commit(user_id, operations)


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


def _stores_with_recording_committer(
    tmp_path,
    queue: InMemoryQueueStore,
) -> tuple[FileSystemSourceStore, InMemoryIndexStore, InMemoryRelationStore, RecordingCommitter]:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()
    committer = RecordingCommitter(
        OperationCommitter(
            source,
            index,
            str(tmp_path),
            relation_store=relations,
            queue_store=queue,
        )
    )
    return source, index, relations, committer


def _context_db(tmp_path, queue: InMemoryQueueStore) -> tuple[ContextDB, RecordingCommitter]:  # noqa: ANN001
    source, index, relations, committer = _stores_with_recording_committer(tmp_path, queue)
    service = SessionCommitService(
        SessionArchiveStore(tmp_path),
        queue,
        committer=cast(Any, committer),
        memory_planner=MemoryCommitPlanner(extractor=RuleFallbackExtractor()),
    )
    database = ContextDB(
        source,
        index,
        relations,
        queue_store=queue,
        session_commit_service=service,
        committer=cast(Any, committer),
    )
    return database, committer


def test_commit_session_async_true_does_not_leave_worker_session_commit_job(tmp_path) -> None:
    queue = InMemoryQueueStore()
    db, committer = _context_db(tmp_path, queue)

    result = db.commit_session(_archive(), async_commit=True)

    assert result.status == "done_with_pending"
    assert result.done is True
    assert result.pending_count > 0
    assert result.pending_persisted is True
    assert result.canonical_active_operation_count == 0
    assert queue.lease("session_commit", 1) == []
    assert committer.calls
    committed_once = sum(len(operations) for _, operations in committer.calls)
    assert committed_once > 0
    assert sum(1 for job in queue.jobs.values() if job.queue_name == "session_commit") == 0
    call_count_after_commit = len(committer.calls)
    assert queue.lease("session_commit", 1) == []
    assert len(committer.calls) == call_count_after_commit


def test_commit_session_async_true_retry_same_task_does_not_recommit_operations(tmp_path) -> None:
    queue = InMemoryQueueStore()
    db, committer = _context_db(tmp_path, queue)
    archive = _archive()

    first = db.commit_session(archive, async_commit=True)
    call_count_after_first = len(committer.calls)
    second = db.commit_session(archive, async_commit=True)

    assert first.task_id == second.task_id
    assert second.status == "done_with_pending"
    assert second.done is True
    assert second.pending_count == first.pending_count
    assert second.pending_persisted is True
    assert len(committer.calls) == call_count_after_first
    assert queue.lease("session_commit", 1) == []


def test_commit_session_async_false_keeps_sync_archive_pending_job(tmp_path) -> None:
    queue = InMemoryQueueStore()
    db, committer = _context_db(tmp_path, queue)

    result = db.commit_session(_archive(), async_commit=False)

    assert result.status == "queued"
    assert result.done is False
    pending = queue.lease("session_commit", 1)
    assert [job.job_id for job in pending] == [result.task_id]
    assert committer.calls == []


def test_session_commit_worker_processes_real_pending_archive_once(tmp_path) -> None:
    queue = InMemoryQueueStore()
    _source, _index, _relations, committer = _stores_with_recording_committer(tmp_path, queue)
    service = SessionCommitService(
        SessionArchiveStore(tmp_path),
        queue,
        committer=cast(Any, committer),
        memory_planner=MemoryCommitPlanner(extractor=RuleFallbackExtractor()),
    )
    archive = _archive()
    queued = service.sync_archive(archive)

    pending = queue.lease("session_commit", 1)
    assert [job.job_id for job in pending] == [queued.task_id]
    result = SessionCommitWorker(service).process_archive(archive)

    assert result == {"task_id": archive.task_id, "status": "done_with_pending", "done": True}
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
        connect_metadata=ConnectMetadata.action_capable_embodied("reachy_mini").to_dict(),
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
        connect_metadata=ConnectMetadata.action_capable_embodied("reachy_mini").to_dict(),
    )

    result = client.process_observation(request, archive_session=True, async_commit=False)

    assert result.prediction_result is not None
    assert result.session_commit_result is not None
    assert result.session_commit_result.status == "queued"
    assert result.session_commit_result.done is False
    assert result.archive_uri == "memoryos://user/u1/sessions/history/ep1"
    pending = queue.lease("session_commit", 1)
    assert [job.job_id for job in pending] == [result.session_commit_result.task_id]


def test_mcp_commit_session_retry_same_payload_enqueues_one_stable_job(tmp_path) -> None:
    queue = InMemoryQueueStore()
    client = MemoryOSClient(str(tmp_path), queue_store=queue)
    server = MemoryOSMCPServer(
        client,
        MCPServerConfig(root=str(tmp_path), user_id="u1", adapter_id="codex", agent_name="codex"),
    )
    args = {"session_id": "s1", "messages": [{"role": "user", "content": "same"}]}

    first = server.call_tool("memoryos_commit_session", args)
    second = server.call_tool("memoryos_commit_session", args)

    assert first["result"]["task_id"] == second["result"]["task_id"]
    assert list(queue.jobs) == [first["result"]["task_id"]]
    assert queue.jobs[first["result"]["task_id"]].status == "pending"
