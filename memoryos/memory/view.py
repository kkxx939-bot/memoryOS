from __future__ import annotations

from typing import Any

from memoryos.memory.schema import MemoryCandidateDraft, MemoryType, MemoryTypeSchema


def project_id_from_archive(archive: Any) -> str:
    metadata = dict(getattr(archive, "metadata", {}) or {})
    for key in ("project_id", "project"):
        value = metadata.get(key)
        if value:
            return str(value)
    connect = metadata.get("connect")
    if isinstance(connect, dict):
        for key in ("project_id", "project"):
            value = connect.get(key)
            if value:
                return str(value)
    for message in getattr(archive, "messages", []) or []:
        if isinstance(message, dict):
            value = message.get("project_id") or message.get("project")
            if value:
                return str(value)
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
        candidate: MemoryCandidateDraft,
        schema: MemoryTypeSchema,
        *,
        user_id: str,
        project_id: str = "",
        adapter_id: str = "",
    ) -> list[str]:
        context = {
            "user_id": user_id,
            "project_id": str(candidate.fields.get("project_id") or project_id or ""),
            "adapter_id": candidate.source_adapter_id or adapter_id or "",
        }
        views = [self._format(template, context) for template in schema.default_retrieval_views]
        views.extend(self._format(view, context) for view in candidate.suggested_retrieval_views)
        if candidate.memory_type == MemoryType.PREFERENCE and context["project_id"]:
            views.append(f"project:{context['project_id']}:knowledge")
        if candidate.memory_type == MemoryType.ENTITY and not context["project_id"]:
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
