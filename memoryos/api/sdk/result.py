from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memoryos.prediction.model.action_result import ActionResult
from memoryos.prediction.model.prediction_result import PredictionResult


@dataclass(frozen=True)
class ProcessObservationResult:
    prediction_result: PredictionResult
    action_result: ActionResult | None = None
    session_commit_result: Any | None = None
    archive_uri: str | None = None

    def to_dict(self) -> dict:
        return {
            "prediction_result": self.prediction_result.to_dict(),
            "action_result": self.action_result.to_dict() if self.action_result is not None else None,
            "session_commit_result": self._session_commit_result_to_dict(),
            "archive_uri": self.archive_uri,
        }

    def _session_commit_result_to_dict(self) -> Any:
        if self.session_commit_result is None:
            return None
        if isinstance(self.session_commit_result, dict):
            return self.session_commit_result
        to_dict = getattr(self.session_commit_result, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        if hasattr(self.session_commit_result, "__dict__"):
            return dict(self.session_commit_result.__dict__)
        return self.session_commit_result
