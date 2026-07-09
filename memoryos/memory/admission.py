from __future__ import annotations

import re
from typing import Any

from memoryos.adapters.agent_hooks.sanitizer import ENV_SECRET_RE, INLINE_SECRET_RE, PRIVATE_KEY_RE, SECRET_KEY_RE
from memoryos.memory.schema import (
    AdmissionDecision,
    MemoryAdmissionResult,
    MemoryCandidateDraft,
    MemoryType,
    MemoryTypeRegistry,
    MemoryTypeSchema,
)
from memoryos.memory.view import MemoryViewRouter

RAW_OUTPUT_RE = re.compile(
    r"(?is)(^diff --git\b|^@@\s|traceback \(most recent call last\)|pytest|failed|error:|stack trace|"
    r"shell output|tool_result|exit code|chunk id:|process exited|^\+\+\+ |^--- )"
)
CHAT_EVENT_RE = re.compile(r"(?i)\b(chat|conversation|discussed|talked about|本次聊天|这次对话|讨论了)\b")
PRIVATE_PROCESS_RE = re.compile(r"(?i)\b(chain of thought|scratchpad|internal reasoning|agent private|内部推理|草稿)\b")


class MemoryAdmissionGate:
    def __init__(self, registry: MemoryTypeRegistry | None = None, view_router: MemoryViewRouter | None = None) -> None:
        self.registry = registry or MemoryTypeRegistry()
        self.view_router = view_router or MemoryViewRouter()

    def evaluate(
        self,
        candidate: MemoryCandidateDraft,
        *,
        user_id: str,
        project_id: str = "",
        adapter_id: str = "",
    ) -> MemoryAdmissionResult:
        schema = self.registry.get(candidate.memory_type)
        text = self._text(candidate)
        if self._is_secret_like(text):
            return self._result(candidate, schema, AdmissionDecision.RESTRICTED, "secret_or_sensitive_content", [], private=True, restricted=True)
        if self._is_private_process(text, candidate):
            private_views = self.view_router.private_view(candidate.source_adapter_id or adapter_id)
            return self._result(candidate, schema, AdmissionDecision.PRIVATE_ONLY, "agent_private_process", private_views, private=True)
        if self._is_raw_output(text, candidate):
            return self._result(candidate, schema, AdmissionDecision.ARCHIVE_ONLY, "raw_tool_or_transient_output", [])
        source_decision = self._source_allowed(candidate, schema)
        if source_decision is not None:
            return self._result(candidate, schema, source_decision, "source_role_not_allowed_by_schema", [])
        missing = [field for field in schema.required_fields if not candidate.fields.get(field)]
        if missing:
            decision = AdmissionDecision.PENDING if candidate.confidence >= 0.65 else AdmissionDecision.REJECT
            return self._result(candidate, schema, decision, f"missing_required_fields:{','.join(missing)}", [])
        typed_decision = self._typed_decision(candidate, project_id=project_id)
        views = self.view_router.route(
            candidate,
            schema,
            user_id=user_id,
            project_id=project_id,
            adapter_id=adapter_id,
        )
        if typed_decision == AdmissionDecision.ACCEPT and not views:
            typed_decision = AdmissionDecision.PENDING
        return self._result(candidate, schema, typed_decision, typed_decision.value, views)

    def _typed_decision(self, candidate: MemoryCandidateDraft, *, project_id: str) -> AdmissionDecision:
        text = self._text(candidate).lower()
        memory_type = candidate.memory_type
        confidence = candidate.confidence
        if memory_type == MemoryType.PROFILE:
            if self._stable_profile(text):
                return AdmissionDecision.ACCEPT if confidence >= 0.75 else AdmissionDecision.PENDING
            return AdmissionDecision.PENDING if confidence >= 0.65 else AdmissionDecision.REJECT
        if memory_type == MemoryType.PREFERENCE:
            if self._stable_preference(text):
                return AdmissionDecision.ACCEPT if confidence >= 0.7 else AdmissionDecision.PENDING
            return AdmissionDecision.PENDING if confidence >= 0.65 else AdmissionDecision.REJECT
        if memory_type == MemoryType.ENTITY:
            return AdmissionDecision.ACCEPT if confidence >= 0.8 else AdmissionDecision.PENDING
        if memory_type == MemoryType.EVENT:
            if CHAT_EVENT_RE.search(text):
                return AdmissionDecision.REJECT
            if self._real_event(text):
                return AdmissionDecision.ACCEPT if confidence >= 0.8 else AdmissionDecision.PENDING
            return AdmissionDecision.PENDING if confidence >= 0.7 else AdmissionDecision.REJECT
        if memory_type == MemoryType.PROJECT_RULE:
            if not (candidate.fields.get("project_id") or project_id):
                return AdmissionDecision.PENDING
            if self._project_rule(text):
                return AdmissionDecision.ACCEPT if confidence >= 0.75 else AdmissionDecision.PENDING
            return AdmissionDecision.PENDING
        if memory_type == MemoryType.PROJECT_DECISION:
            if not (candidate.fields.get("project_id") or project_id):
                return AdmissionDecision.PENDING
            if self._project_decision(text):
                return AdmissionDecision.ACCEPT if confidence >= 0.75 else AdmissionDecision.PENDING
            return AdmissionDecision.PENDING
        if memory_type == MemoryType.AGENT_EXPERIENCE:
            if self._agent_experience(candidate):
                return AdmissionDecision.ACCEPT if confidence >= 0.78 else AdmissionDecision.PENDING
            return AdmissionDecision.REJECT
        return AdmissionDecision.REJECT

    def _source_allowed(self, candidate: MemoryCandidateDraft, schema: MemoryTypeSchema) -> AdmissionDecision | None:
        role = candidate.source_role.lower()
        if role == "tool" and not schema.allow_tool_source:
            return AdmissionDecision.ARCHIVE_ONLY
        if role in {"assistant", "agent"} and not schema.allow_assistant_source:
            return AdmissionDecision.PENDING
        if role == "user" and not schema.allow_user_source:
            return AdmissionDecision.REJECT
        return None

    def _result(
        self,
        candidate: MemoryCandidateDraft,
        schema: MemoryTypeSchema,
        decision: AdmissionDecision,
        reason: str,
        views: list[str],
        *,
        private: bool = False,
        restricted: bool = False,
    ) -> MemoryAdmissionResult:
        return MemoryAdmissionResult(
            decision=decision,
            reason=reason,
            confidence=candidate.confidence,
            memory_type=candidate.memory_type,
            retrieval_views=views,
            operation_mode=schema.operation_mode,
            merge_key=candidate.merge_key,
            private=private,
            restricted=restricted,
        )

    def _text(self, candidate: MemoryCandidateDraft) -> str:
        evidence = " ".join(str(item) for item in candidate.evidence)
        fields = " ".join(str(value) for value in candidate.fields.values())
        return "\n".join([candidate.title, candidate.content, fields, evidence])

    def _is_secret_like(self, text: str) -> bool:
        return bool(
            PRIVATE_KEY_RE.search(text)
            or ENV_SECRET_RE.search(text)
            or INLINE_SECRET_RE.search(text)
            or ("<redacted" in text.lower() and SECRET_KEY_RE.search(text))
            or re.search(r"(?i)\b(authorization\s*:|cookie\s*:)", text)
        )

    def _is_raw_output(self, text: str, candidate: MemoryCandidateDraft) -> bool:
        role = candidate.source_role.lower()
        if role == "tool":
            return True
        return bool(RAW_OUTPUT_RE.search(text))

    def _is_private_process(self, text: str, candidate: MemoryCandidateDraft) -> bool:
        return candidate.source_role.lower() in {"agent_private", "internal"} or bool(PRIVATE_PROCESS_RE.search(text))

    def _stable_profile(self, text: str) -> bool:
        return any(token in text for token in ("i am ", "i work", "我是", "我在", "长期", "工作于", "负责人"))

    def _stable_preference(self, text: str) -> bool:
        return any(token in text for token in ("prefer", "preference", "i like", "i dislike", "我喜欢", "我不喜欢", "偏好", "习惯", "沟通", "代码审查", "以后请"))

    def _real_event(self, text: str) -> bool:
        return any(token in text for token in ("completed", "implemented", "fixed", "released", "decided", "完成", "修复", "发布", "已采用", "决定"))

    def _project_rule(self, text: str) -> bool:
        return any(token in text for token in ("must", "never", "do not", "禁止", "不允许", "必须", "不要", "项目规则", "约束", "最高强约束"))

    def _project_decision(self, text: str) -> bool:
        return any(token in text for token in ("decided", "adopted", "deferred", "rejected", "决定", "采用", "暂缓", "不做", "取舍", "架构决策"))

    def _agent_experience(self, candidate: MemoryCandidateDraft) -> bool:
        fields = candidate.fields
        return bool(fields.get("situation") and fields.get("approach") and fields.get("outcome"))


class MemoryEmbeddingBackend:
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class MemoryRerankerBackend:
    def rerank(self, query: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raise NotImplementedError
