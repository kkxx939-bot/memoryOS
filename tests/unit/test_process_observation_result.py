from __future__ import annotations

from typing import Any

import pytest

from memoryos.action_policy.model.action_policy import ActionCandidate
from memoryos.api.sdk import ProcessObservationResult
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.behavior.model.observation import Observation
from memoryos.contextdb.session.session_model import SessionArchive, SessionCommitResult
from memoryos.prediction.model.action_context import ActionContext
from memoryos.prediction.model.action_result import ActionResult
from memoryos.prediction.model.prediction_request import PredictionRequest
from memoryos.prediction.model.prediction_result import PolicyDecision, PredictionResult


class FakeEngine:
    def __init__(self, result: PredictionResult) -> None:
        self.result = result
        self.calls: list[tuple[PredictionRequest, list | None]] = []

    def process(self, request: PredictionRequest, policies: list | None = None) -> PredictionResult:
        self.calls.append((request, policies))
        return self.result


class FakeExecutor:
    def __init__(self, result: ActionResult) -> None:
        self.result = result
        self.calls: list[tuple[PolicyDecision, ActionContext]] = []

    def execute(self, decision: PolicyDecision, action_context: ActionContext) -> ActionResult:
        self.calls.append((decision, action_context))
        return self.result


class FakeContextDB:
    def __init__(self, commit_result: Any) -> None:
        self.commit_result = commit_result
        self.calls: list[tuple[SessionArchive, bool]] = []

    def commit_session(self, archive: SessionArchive, *, async_commit: bool = True) -> Any:
        self.calls.append((archive, async_commit))
        return self.commit_result


class DictLikeCommitResult:
    def to_dict(self) -> dict:
        return {"status": "done", "source": "to_dict"}


def _prediction_result() -> PredictionResult:
    observation = Observation(user_id="u1", raw_text="hot room", location="home")
    action_context = ActionContext(
        user_id="u1",
        candidate_actions=["turn_on_ac"],
        packed_context={"load_plan": []},
        source_uris=["memoryos://skills/ac"],
    )
    return PredictionResult(
        request_id="req1",
        episode_id="s1",
        observation=observation,
        candidates=[
            ActionCandidate(
                action="turn_on_ac",
                score=0.9,
                policy_uri="memoryos://user/u1/action_policies/home/turn_on_ac",
                reason="hot room",
            )
        ],
        action_context=action_context,
        decision=PolicyDecision(mode="execute", allowed=True, action="turn_on_ac", reason="ok"),
    )


def _action_result() -> ActionResult:
    return ActionResult(
        action="turn_on_ac",
        status="success",
        executed=True,
        reason="ok",
        tool_name="ac_tool",
        output={"device": "on"},
    )


def _request(*, session_uri: str = "") -> PredictionRequest:
    return PredictionRequest(
        user_id="u1",
        episode_id="s1",
        observation="hot room",
        available_actions=["turn_on_ac", "ask_user", "do_nothing"],
        request_id="req1",
        session_uri=session_uri,
    )


def _client(
    prediction_result: PredictionResult | None = None,
    action_result: ActionResult | None = None,
    commit_result: Any | None = None,
) -> MemoryOSClient:
    client = object.__new__(MemoryOSClient)
    client.engine = FakeEngine(prediction_result or _prediction_result())
    client.executor = FakeExecutor(action_result or _action_result())
    client.context_db = FakeContextDB(commit_result)
    return client


def test_process_observation_returns_outer_result_with_prediction_action_commit_and_archive_uri() -> None:
    prediction_result = _prediction_result()
    action_result = _action_result()
    commit_result = {"status": "done", "archive_uri": "memoryos://custom/session"}
    client = _client(prediction_result, action_result, commit_result)

    result = client.process_observation(
        _request(session_uri="memoryos://custom/session"),
        archive_session=True,
        async_commit=False,
    )

    assert isinstance(result, ProcessObservationResult)
    assert result.prediction_result is prediction_result
    assert result.action_result is action_result
    assert result.session_commit_result == commit_result
    assert result.archive_uri == "memoryos://custom/session"
    assert result.to_dict()["prediction_result"] == prediction_result.to_dict()
    assert result.to_dict()["action_result"] == action_result.to_dict()


def test_process_observation_without_archive_executes_action_and_does_not_commit() -> None:
    client = _client()

    result = client.process_observation(_request(), archive_session=False)

    assert isinstance(result, ProcessObservationResult)
    assert len(client.engine.calls) == 1
    assert len(client.executor.calls) == 1
    assert client.context_db.calls == []
    assert result.prediction_result is client.engine.result
    assert result.action_result is client.executor.result
    assert result.session_commit_result is None
    assert result.archive_uri is None


def test_process_observation_archive_commit_result_and_async_flag_are_visible() -> None:
    commit_result = SessionCommitResult(task_id="task1", archive_uri="memoryos://archive/s1", status="queued")
    client = _client(commit_result=commit_result)

    result = client.process_observation(_request(session_uri="memoryos://archive/s1"), archive_session=True, async_commit=False)

    assert len(client.context_db.calls) == 1
    archive, async_commit = client.context_db.calls[0]
    assert async_commit is False
    assert result.session_commit_result is commit_result
    assert result.archive_uri == archive.archive_uri == "memoryos://archive/s1"


def test_process_observation_commit_status_is_visible_for_async_modes() -> None:
    queued = SessionCommitResult(task_id="task1", archive_uri="memoryos://archive/s1", status="queued")
    done = SessionCommitResult(task_id="task2", archive_uri="memoryos://archive/s2", status="done", done=True)

    queued_result = _client(commit_result=queued).process_observation(
        _request(session_uri="memoryos://archive/s1"),
        archive_session=True,
        async_commit=False,
    )
    done_result = _client(commit_result=done).process_observation(
        _request(session_uri="memoryos://archive/s2"),
        archive_session=True,
        async_commit=True,
    )

    assert queued_result.prediction_result is not None
    assert queued_result.session_commit_result.status == "queued"
    assert queued_result.archive_uri == "memoryos://archive/s1"
    assert done_result.prediction_result is not None
    assert done_result.session_commit_result.status == "done"
    assert done_result.archive_uri == "memoryos://archive/s2"


def test_process_observation_archive_passes_async_true() -> None:
    client = _client(commit_result={"status": "done"})

    client.process_observation(_request(), archive_session=True, async_commit=True)

    assert client.context_db.calls[0][1] is True


def test_predict_still_returns_prediction_result_without_action_or_commit() -> None:
    prediction_result = _prediction_result()
    client = _client(prediction_result=prediction_result)

    result = client.predict(_request())

    assert result is prediction_result
    assert isinstance(result, PredictionResult)
    assert not isinstance(result, ProcessObservationResult)
    assert len(client.engine.calls) == 1
    assert client.executor.calls == []
    assert client.context_db.calls == []


def test_commit_agent_session_returns_context_db_commit_result() -> None:
    commit_result = {"status": "done", "archive_uri": "memoryos://user/u1/sessions/history/s1"}
    client = _client(commit_result=commit_result)

    result = client.commit_agent_session(
        user_id="u1",
        session_id="s1",
        messages=[{"role": "user", "content": "remember this"}],
        async_commit=False,
    )

    assert result == commit_result
    assert client.context_db.calls[0][1] is False


def test_process_observation_result_to_dict_handles_optional_and_commit_shapes() -> None:
    prediction_result = _prediction_result()

    assert ProcessObservationResult(prediction_result=prediction_result).to_dict() == {
        "prediction_result": prediction_result.to_dict(),
        "action_result": None,
        "session_commit_result": None,
        "archive_uri": None,
    }
    assert ProcessObservationResult(
        prediction_result=prediction_result,
        session_commit_result={"status": "queued"},
    ).to_dict()["session_commit_result"] == {"status": "queued"}
    assert ProcessObservationResult(
        prediction_result=prediction_result,
        session_commit_result=DictLikeCommitResult(),
    ).to_dict()["session_commit_result"] == {"status": "done", "source": "to_dict"}


def test_prediction_result_still_rejects_memory_operations() -> None:
    base = _prediction_result()

    with pytest.raises(ValueError, match="durable memory operations"):
        PredictionResult(
            request_id=base.request_id,
            episode_id=base.episode_id,
            observation=base.observation,
            candidates=base.candidates,
            action_context=base.action_context,
            decision=base.decision,
            memory_operations=[{"op": "write"}],
        )
