from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

from memoryos.application.prediction.observation_processor import PredictiveObservationProcessor
from memoryos.application.session.commit_service import SessionCommitService
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.prediction.model.prediction_request import PredictionRequest
from memoryos.prediction.pipeline.prediction_engine import PredictionEngine


def _request(*, session_uri: str = "") -> PredictionRequest:
    return PredictionRequest(
        user_id="user-1",
        episode_id="episode-1",
        observation="hot room",
        available_actions=["turn_on_ac"],
        session_uri=session_uri,
    )


def _prediction() -> Any:
    prediction = Mock()
    prediction.observation = SimpleNamespace(raw_text="hot room")
    prediction.action_context = SimpleNamespace(source_uris=["memoryos://user/user-1/memories/memory-1"])
    prediction.to_dict.return_value = {"request_id": "request-1"}
    return prediction


def test_processor_delegates_prediction_without_archiving_when_disabled() -> None:
    engine = Mock()
    session_commit_service = Mock()
    prediction = _prediction()
    policies: list[Any] = []
    request = _request()
    engine.process.return_value = prediction

    processor = PredictiveObservationProcessor(
        cast(PredictionEngine, engine),
        cast(SessionCommitService, session_commit_service),
    )

    assert processor.process(request, policies, archive_session=False) is prediction
    engine.process.assert_called_once_with(request, policies)
    session_commit_service.sync_archive.assert_not_called()


def test_processor_archives_prediction_with_stable_session_payload() -> None:
    engine = Mock()
    session_commit_service = Mock()
    prediction = _prediction()
    request = _request(session_uri="memoryos://user/user-1/sessions/custom/")
    engine.process.return_value = prediction

    processor = PredictiveObservationProcessor(
        cast(PredictionEngine, engine),
        cast(SessionCommitService, session_commit_service),
    )

    assert processor.process(request, []) is prediction
    archive = session_commit_service.sync_archive.call_args.args[0]
    assert isinstance(archive, SessionArchive)
    assert archive.user_id == "user-1"
    assert archive.session_id == "episode-1"
    assert archive.archive_uri == "memoryos://user/user-1/sessions/custom/"
    assert archive.observations == [{"raw_text": "hot room"}]
    assert archive.predictions == [{"request_id": "request-1"}]
    assert archive.used_contexts == [{"uri": "memoryos://user/user-1/memories/memory-1"}]
