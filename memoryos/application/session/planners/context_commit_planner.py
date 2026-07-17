"""Generic context planning within the session application flow."""

from __future__ import annotations

from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class ContextCommitPlanner:
    def plan(self, archive: SessionArchive) -> list[ContextOperation]:
        operations: list[ContextOperation] = []
        seen: set[str] = set()
        for item in archive.used_contexts:
            uri = str(item.get("uri", ""))
            if not uri or uri in seen:
                continue
            if item.get("refresh_layers") is False:
                continue
            seen.add(uri)
            context_type = ContextType(str(item.get("context_type", self._infer_type(uri))))
            operations.append(
                ContextOperation(
                    user_id=archive.user_id,
                    context_type=context_type,
                    action=OperationAction.REFRESH_LAYERS,
                    target_uri=uri,
                    payload={"reason": "session_commit_refresh"},
                    evidence=[{"task_id": archive.task_id}],
                    source_session_id=archive.session_id,
                )
            )
        for observation in archive.observations:
            case_uri = observation.get("case_uri")
            if case_uri:
                operations.append(
                    ContextOperation(
                        user_id=archive.user_id,
                        context_type=ContextType.BEHAVIOR_CASE,
                        action=OperationAction.ARCHIVE,
                        target_uri=str(case_uri),
                        payload={"reason": "session_commit_archive"},
                        evidence=[{"task_id": archive.task_id}],
                        source_session_id=archive.session_id,
                    )
                )
        return operations

    def _infer_type(self, uri: str) -> str:
        if "/action_policies/" in uri:
            return ContextType.ACTION_POLICY.value
        if "/behavior/patterns/" in uri:
            return ContextType.BEHAVIOR_PATTERN.value
        if "/behavior/cases/" in uri:
            return ContextType.BEHAVIOR_CASE.value
        if uri.startswith("memoryos://resources/"):
            return ContextType.RESOURCE.value
        if uri.startswith("memoryos://skills/"):
            return ContextType.SKILL.value
        return ContextType.MEMORY.value
