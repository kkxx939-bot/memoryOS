"""ActionPolicy 在线决策、动作执行与会话归档工作流。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from foundation.readiness import RuntimeReadiness
from policy.action_policy.decision.request import PredictionRequest
from policy.action_policy.decision.result import PredictionResult
from policy.action_policy.execution.result import ActionResult
from policy.action_policy.feedback import build_action_feedback
from policy.action_policy.model.action_policy import ActionPolicy
from policy.action_policy.workflow.result import ProcessObservationResult
from pre.connect import ConnectMetadata
from pre.session import SessionArchive


def _uri_items(uris: list[str]) -> list[dict[str, str]]:
    """把 URI 列表去重后转换为会话归档条目。"""

    return [{"uri": uri} for uri in dict.fromkeys(str(uri) for uri in uris if uri)]


def _merge_uri_items(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 URI 合并上下文条目，并保留首次出现时的附加字段。"""

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            uri = str(item.get("uri", ""))
            if not uri or uri in seen:
                continue
            seen.add(uri)
            merged.append(dict(item))
    return merged


class ActionPolicyWorkflowService:
    """协调策略引擎、动作执行器和会话提交，不实现三者内部规则。"""

    def __init__(
        self,
        *,
        tenant_id: str,
        engine: Any,
        executor: Any,
        session_commit_service: Any,
        readiness: RuntimeReadiness | None,
        require_predict_metadata: Callable[[dict[str, Any] | None], ConnectMetadata],
        require_process_observation_metadata: Callable[[dict[str, Any] | None], ConnectMetadata],
    ) -> None:
        self.tenant_id = tenant_id
        self.engine = engine
        self.executor = executor
        self.session_commit_service = session_commit_service
        self._readiness = readiness
        self._require_predict_metadata = require_predict_metadata
        self._require_process_observation_metadata = require_process_observation_metadata

    def _require_ready(self) -> None:
        if self._readiness is not None:
            self._readiness.require_ready()

    def predict(self, request: PredictionRequest, policies: list[ActionPolicy] | None = None) -> PredictionResult:
        """执行不产生外部动作的在线策略预测。"""

        self._require_ready()
        self._require_predict_metadata(request.connect_metadata)
        return self.engine.process(request, policies=policies)

    def process_observation(
        self,
        request: PredictionRequest,
        policies: list[ActionPolicy] | None = None,
        *,
        archive_session: bool = True,
        async_commit: bool = True,
    ) -> ProcessObservationResult:
        """协调观察决策、动作执行、反馈形成和可选会话归档。"""

        self._require_ready()
        metadata = self._require_process_observation_metadata(request.connect_metadata)
        connect_metadata = metadata.to_dict()
        result = self.engine.process(request, policies=policies)
        try:
            action_result = self.executor.execute(result.decision, result.action_context)
        except Exception as exc:
            action_result = ActionResult(
                action=result.decision.action,
                status="failed",
                executed=False,
                reason="ActionExecutor raised",
                error=exc.__class__.__name__,
            )
        if not archive_session:
            return ProcessObservationResult(
                prediction_result=result,
                action_result=action_result,
                session_commit_result=None,
                archive_uri=None,
            )
        policy_uri = result.candidates[0].policy_uri if result.candidates else ""
        feedback = []
        if action_result.status in {"success", "failed", "blocked"} and policy_uri:
            feedback.append(
                build_action_feedback(
                    action_result,
                    user_id=request.user_id,
                    episode_id=request.episode_id,
                    policy_uri=policy_uri,
                    scene_key=result.observation.scene_key,
                )
            )
        observation_payload = {
            **result.observation.__dict__,
            "episode_id": request.episode_id,
            "request_id": request.request_id or result.request_id,
            "scene_key": result.observation.scene_key,
        }
        used_contexts = _merge_uri_items(
            [{"uri": uri} for uri in result.action_context.source_uris],
            [{"uri": uri, "refresh_layers": False} for uri in action_result.resource_uris],
        )
        used_skills = _uri_items(
            [
                *[uri for uri in result.action_context.source_uris if uri.startswith("memoryos://skills/")],
                *action_result.skill_uris,
            ]
        )
        archive = SessionArchive(
            user_id=request.user_id,
            session_id=request.episode_id,
            archive_uri=request.session_uri
            or f"memoryos://user/{request.user_id}/sessions/history/{request.episode_id}",
            observations=[observation_payload],
            predictions=[result.to_dict()],
            action_results=[
                {
                    "request_id": result.request_id,
                    "episode_id": result.episode_id,
                    "decision": result.decision.to_dict(),
                    "selected_action": result.decision.action,
                    "action_result": action_result.to_dict(),
                }
            ],
            feedback=feedback,
            used_contexts=used_contexts,
            used_skills=used_skills,
            metadata={
                "connect": connect_metadata,
                "tenant_id": self.tenant_id,
            },
        )
        archive_error = None
        try:
            commit_result = self.session_commit_service.commit_session(archive, async_commit=async_commit)
        except Exception as exc:
            commit_result = None
            archive_error = {"code": "ARCHIVE_COMMIT_FAILED", "message": exc.__class__.__name__}
        return ProcessObservationResult(
            prediction_result=result,
            action_result=action_result,
            session_commit_result=commit_result,
            archive_uri=archive.archive_uri,
            archive_error=archive_error,
        )


__all__ = ["ActionPolicyWorkflowService"]
