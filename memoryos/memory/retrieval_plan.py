"""记忆系统里的检索计划。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MemoryRetrievalPlan:
    search_scope: str = "default"
    retrieval_views: list[str] = field(default_factory=list)
    include_candidates: bool = False


class MemoryRetrievalPlanner:
    PROJECT_SCOPES = {
        "project_rules": "rules",
        "project_decisions": "decisions",
        "project_knowledge": "knowledge",
        "project_agent_experience": "agent_experience",
    }

    def build(
        self,
        *,
        user_id: str | None,
        adapter_id: str = "",
        project_id: str = "",
        search_scope: str | None = None,
        retrieval_views: list[str] | None = None,
    ) -> MemoryRetrievalPlan:
        explicit_views = [str(view) for view in (retrieval_views or []) if str(view)]
        scope = str(search_scope or "default")
        include_candidates = scope in {"candidates", "all_with_candidates"}
        if explicit_views:
            return MemoryRetrievalPlan(scope, list(dict.fromkeys(explicit_views)), include_candidates=include_candidates)
        views = self._scope_views(scope, user_id=user_id or "", adapter_id=adapter_id, project_id=project_id)
        return MemoryRetrievalPlan(scope, views, include_candidates=include_candidates)

    def _scope_views(self, scope: str, *, user_id: str, adapter_id: str, project_id: str) -> list[str]:
        if scope == "agent_private":
            return [f"agent:{adapter_id}:private"] if adapter_id else []
        if scope == "user_profile":
            return [f"user:{user_id}:profile"] if user_id else []
        if scope == "user_preferences":
            return [f"user:{user_id}:preferences"] if user_id else []
        if scope in self.PROJECT_SCOPES:
            return [f"project:{project_id}:{self.PROJECT_SCOPES[scope]}"] if project_id else []
        if scope == "all_shared_memory":
            return self._shared_views(user_id=user_id, project_id=project_id)
        if scope in {"candidates", "all_with_candidates"}:
            return self._default_views(user_id=user_id, adapter_id=adapter_id, project_id=project_id)
        return self._default_views(user_id=user_id, adapter_id=adapter_id, project_id=project_id)

    def _default_views(self, *, user_id: str, adapter_id: str, project_id: str) -> list[str]:
        views = []
        if adapter_id:
            views.append(f"agent:{adapter_id}:private")
        views.extend(self._shared_views(user_id=user_id, project_id=project_id))
        return list(dict.fromkeys(views))

    def _shared_views(self, *, user_id: str, project_id: str) -> list[str]:
        views = []
        if user_id:
            views.extend([f"user:{user_id}:profile", f"user:{user_id}:preferences"])
        if project_id:
            views.extend(
                [
                    f"project:{project_id}:rules",
                    f"project:{project_id}:decisions",
                    f"project:{project_id}:knowledge",
                    f"project:{project_id}:agent_experience",
                ]
            )
        return views
