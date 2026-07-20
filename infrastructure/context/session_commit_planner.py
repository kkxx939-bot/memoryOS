"""根据 Session 使用记录规划 Context 分层刷新。"""

from __future__ import annotations

from infrastructure.store.model.context.context_type import ContextType
from pre.session import SessionArchive
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_action import OperationAction


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
            declared_type = item.get("context_type")
            inferred_type = self._infer_type(uri)
            if declared_type in (None, "") and inferred_type is None:
                # 未声明类型的 URI 仍保留在不可变 Session 证据中，
                # 但它不能授权修改普通 Context Source。
                continue
            context_type = ContextType(str(declared_type or inferred_type))
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
        return operations

    def _infer_type(self, uri: str) -> str | None:
        if "/action_policies/" in uri:
            return ContextType.ACTION_POLICY.value
        if "/support/behavior/" in uri:
            return ContextType.BEHAVIOR_SUPPORT.value
        if "/support/action_policy/" in uri:
            return ContextType.ACTION_POLICY_SUPPORT.value
        if "/behavior/patterns/" in uri:
            return ContextType.BEHAVIOR_PATTERN.value
        if "/behavior/clusters/" in uri:
            return ContextType.BEHAVIOR_CLUSTER.value
        if "/behavior/cases/" in uri:
            return ContextType.BEHAVIOR_CASE.value
        if uri.startswith("memoryos://resources/") or "/resources/" in uri:
            return ContextType.RESOURCE.value
        if uri.startswith("memoryos://skills/") or "/skills/" in uri:
            return ContextType.SKILL.value
        return None
