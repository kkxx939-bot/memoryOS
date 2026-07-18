from __future__ import annotations

from memoryos.api.mcp.config import MCPServerConfig
from memoryos.api.mcp.server import MemoryOSMCPServer
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.connect import ConnectMetadata
from memoryos.contextdb.store.local_stores import InMemoryQueueStore
from memoryos.prediction.model.prediction_request import PredictionRequest


def _request() -> PredictionRequest:
    return PredictionRequest(
        user_id="u1",
        episode_id="ep1",
        observation="hot room",
        available_actions=["turn_on_ac", "ask_user", "do_nothing"],
        request_id="req1",
        connect_metadata=ConnectMetadata.action_capable_embodied("reachy_mini").to_dict(),
    )


def test_process_observation_inline_commit_leaves_no_session_job(tmp_path) -> None:  # noqa: ANN001
    queue = InMemoryQueueStore()
    client = MemoryOSClient(str(tmp_path), queue_store=queue)

    result = client.process_observation(_request(), archive_session=True, async_commit=True)

    assert result.session_commit_result is not None
    assert result.session_commit_result.status == "done"
    assert result.session_commit_result.done is True
    assert result.archive_uri == "memoryos://user/u1/sessions/history/ep1"
    assert queue.lease("session_commit", lease_owner="test", limit=1) == []


def test_process_observation_deferred_commit_enqueues_exact_session_job(tmp_path) -> None:  # noqa: ANN001
    queue = InMemoryQueueStore()
    client = MemoryOSClient(str(tmp_path), queue_store=queue)

    result = client.process_observation(_request(), archive_session=True, async_commit=False)

    assert result.prediction_result is not None
    assert result.session_commit_result is not None
    assert result.session_commit_result.status == "queued"
    assert result.session_commit_result.done is False
    assert result.archive_uri == "memoryos://user/u1/sessions/history/ep1"
    pending = queue.lease("session_commit", lease_owner="test", limit=1)
    assert [job.job_id for job in pending] == [result.session_commit_result.task_id]


def test_mcp_commit_session_retry_same_payload_enqueues_one_stable_job(tmp_path) -> None:  # noqa: ANN001
    queue = InMemoryQueueStore()
    client = MemoryOSClient(str(tmp_path), queue_store=queue)
    server = MemoryOSMCPServer(
        client,
        MCPServerConfig(
            root=str(tmp_path),
            user_id="u1",
            adapter_id="codex",
            agent_name="codex",
            allowed_workspace_ids=frozenset({"project-a"}),
        ),
    )
    args = {
        "session_id": "s1",
        "project_id": "project-a",
        "messages": [{"role": "user", "content": "same"}],
    }

    first = server.call_tool("memoryos_commit_session", args)
    second = server.call_tool("memoryos_commit_session", args)

    assert first["result"]["task_id"] == second["result"]["task_id"]
    assert list(queue.jobs) == [first["result"]["task_id"]]
    assert queue.jobs[first["result"]["task_id"]].status == "pending"
