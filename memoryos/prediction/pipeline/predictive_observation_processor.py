from __future__ import annotations

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.contextdb.session.session_commit import SessionCommitService
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.prediction.model.prediction_request import PredictionRequest
from memoryos.prediction.model.prediction_result import PredictionResult
from memoryos.prediction.pipeline.prediction_engine import PredictionEngine


class PredictiveObservationProcessor:
    """Production observation entrypoint for the predictive pipeline."""

    def __init__(self, prediction_engine: PredictionEngine, session_commit_service: SessionCommitService | None = None) -> None:
        self.prediction_engine = prediction_engine
        self.session_commit_service = session_commit_service

    def process(
        self,
        request: PredictionRequest,
        policies: list[ActionPolicy],
        archive_session: bool = True,
    ) -> PredictionResult:
        prediction = self.prediction_engine.process(request, policies)
        if archive_session and self.session_commit_service is not None:
            archive_uri = request.session_uri or f"memoryos://user/{request.user_id}/sessions/history/{request.episode_id}/"
            self.session_commit_service.sync_archive(
                SessionArchive(
                    user_id=request.user_id,
                    session_id=request.episode_id,
                    archive_uri=archive_uri,
                    observations=[prediction.observation.__dict__],
                    predictions=[prediction.to_dict()],
                    used_contexts=[{"uri": uri} for uri in prediction.action_context.source_uris],
                )
            )
        return prediction
