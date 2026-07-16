"""记忆视图的路由规则。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from memoryos.memory.canonical.proposal import MemorySemanticProposal
from memoryos.memory.schema import MemoryCandidateDraft, MemoryType, MemoryTypeSchema
from memoryos.security.workspace_identity import normalize_workspace_id, repository_workspace_id


def project_id_from_archive(archive: Any) -> str:
    metadata = dict(getattr(archive, "metadata", {}) or {})
    for key in ("project_id", "project"):
        value = metadata.get(key)
        if value:
            return normalize_workspace_id(value)
    connect = metadata.get("connect")
    if isinstance(connect, dict):
        for key in ("project_id", "project"):
            value = connect.get(key)
            if value:
                return normalize_workspace_id(value)
        extra = connect.get("extra")
        if isinstance(extra, dict):
            for key in ("project_id", "project"):
                value = extra.get(key)
                if value:
                    return normalize_workspace_id(value)
            if extra.get("repo"):
                return repository_workspace_id(
                    repo_root=extra["repo"],
                    cwd=extra.get("cwd") or "",
                    git_remote=extra.get("git_remote") or "",
                )
    for message in getattr(archive, "messages", []) or []:
        if isinstance(message, dict):
            value = message.get("project_id") or message.get("project")
            if value:
                return normalize_workspace_id(value)
    return ""


def adapter_id_from_archive(archive: Any) -> str:
    metadata = dict(getattr(archive, "metadata", {}) or {})
    connect = metadata.get("connect")
    if isinstance(connect, dict) and connect.get("adapter_id"):
        return str(connect["adapter_id"])
    if metadata.get("adapter_id"):
        return str(metadata["adapter_id"])
    return ""


class MemoryViewRouter:
    def route(
        self,
        candidate: MemoryCandidateDraft | MemorySemanticProposal,
        schema: MemoryTypeSchema,
        *,
        user_id: str,
        project_id: str = "",
        adapter_id: str = "",
    ) -> list[str]:
        suggested_views: Sequence[Any]
        if isinstance(candidate, MemorySemanticProposal):
            fields = {**dict(candidate.identity_fields), **dict(candidate.value_fields)}
            source_adapter_id = str(candidate.metadata.get("source_adapter_id") or "")
            suggested_views = tuple(candidate.metadata.get("suggested_retrieval_views", ()) or ())
            memory_type = MemoryType(candidate.memory_type)
        else:
            fields = candidate.fields
            source_adapter_id = candidate.source_adapter_id
            suggested_views = candidate.suggested_retrieval_views
            memory_type = candidate.memory_type
        context = {
            "user_id": user_id,
            "project_id": str(fields.get("project_id") or project_id or ""),
            "adapter_id": source_adapter_id or adapter_id or "",
        }
        views = [self._format(template, context) for template in schema.default_retrieval_views]
        views.extend(self._format(str(view), context) for view in suggested_views)
        if memory_type == MemoryType.PREFERENCE and context["project_id"]:
            views.append(f"project:{context['project_id']}:knowledge")
        if memory_type == MemoryType.ENTITY and not context["project_id"]:
            views.append(f"user:{user_id}:profile")
        if not schema.share_default and context["adapter_id"]:
            views = [f"agent:{context['adapter_id']}:private"]
        return list(dict.fromkeys(view for view in views if view))

    def private_view(self, adapter_id: str) -> list[str]:
        return [f"agent:{adapter_id}:private"] if adapter_id else []

    def _format(self, template: str, context: dict[str, str]) -> str:
        if "{project_id}" in template and not context.get("project_id"):
            return ""
        if "{adapter_id}" in template and not context.get("adapter_id"):
            return ""
        return template.format(**context)
