"""观察处理工作流的组合结果。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from policy.action_policy.decision.result import PredictionResult
from policy.action_policy.execution.result import ActionResult


@dataclass(frozen=True)
class ProcessObservationResult:
    """汇总策略决策、动作执行和可选会话提交结果。"""

    prediction_result: PredictionResult
    action_result: ActionResult | None = None
    session_commit_result: Any | None = None
    archive_uri: str | None = None
    archive_error: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        """转换为对外接口可以直接序列化的字典。"""

        return {
            "prediction_result": self.prediction_result.to_dict(),
            "action_result": self.action_result.to_dict() if self.action_result is not None else None,
            "session_commit_result": self._session_commit_result_to_dict(),
            "archive_uri": self.archive_uri,
            "archive_error": self.archive_error,
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
