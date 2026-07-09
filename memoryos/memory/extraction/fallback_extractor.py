from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.core.ids import stable_hash
from memoryos.memory.extraction.memory_extractor import MemoryExtractorBackend
from memoryos.memory.schema import MemoryCandidateDraft, MemoryType, MemoryTypeSchema
from memoryos.memory.view import adapter_id_from_archive, project_id_from_archive

RULE_RE = re.compile(r"(?i)(must|never|do not|don't|禁止|不允许|必须|不要|以后别|项目规则|最高强约束|约束)")
PREFERENCE_RE = re.compile(r"(?i)(prefer|preference|remember|i like|i dislike|我喜欢|我不喜欢|偏好|习惯|沟通偏好|代码审查偏好|以后请)")
PROFILE_RE = re.compile(r"(?i)(^i am\b|i work\b|我是|我在|负责人|长期从事)")
DECISION_RE = re.compile(r"(?i)(decided|adopted|deferred|rejected|决定|采用|暂缓|不做|架构决策|阶段性取舍)")
EVENT_RE = re.compile(r"(?i)(completed|implemented|fixed|released|verified|完成|修复|发布|已验证|已实现)")
AGENT_EXPERIENCE_RE = re.compile(r"(?i)(reusable|lesson|pattern|approach|outcome|verified|implemented|fixed|经验|可复用|做法|结果|验证)")
ENTITY_RE = re.compile(r"(?i)(project|tool|product|organization|person|device|concept|项目|工具|产品|组织|人物|设备|概念)[：:\s]+([\w./@-]+)")
REMEMBER_MARKERS = ("记住：", "记住:", "remember:", "Remember:")


class RuleFallbackExtractor(MemoryExtractorBackend):
    """Deterministic candidate extractor.

    This is intentionally not an operation writer. It emits structured drafts
    that must pass schema admission before MemoryCommitPlanner can build ops.
    """

    def extract(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
    ) -> list[MemoryCandidateDraft]:
        schema_types = {schema.memory_type for schema in schemas} or set(MemoryType)
        candidates: list[MemoryCandidateDraft] = []
        project_id = project_id_from_archive(archive)
        adapter_id = adapter_id_from_archive(archive)
        for index, message in enumerate(archive.messages):
            if not isinstance(message, dict):
                continue
            text = str(message.get("content", message.get("text", ""))).strip()
            if not text:
                continue
            role = str(message.get("role", "user") or "user")
            evidence_id = str(message.get("id") or message.get("message_id") or f"message:{index}")
            candidates.extend(
                self._message_candidates(
                    text,
                    role=role,
                    adapter_id=adapter_id,
                    session_id=archive.session_id,
                    evidence_id=evidence_id,
                    project_id=project_id,
                    schema_types=schema_types,
                )
            )
        for index, tool_result in enumerate(archive.tool_results):
            content = json.dumps(tool_result, ensure_ascii=False, sort_keys=True)
            candidates.append(
                self._candidate(
                    MemoryType.EVENT,
                    title="Tool result archive evidence",
                    content=content,
                    fields={"event": "tool_result", "project_id": project_id},
                    confidence=0.25,
                    role="tool",
                    adapter_id=adapter_id,
                    session_id=archive.session_id,
                    evidence_id=f"tool_result:{index}",
                    reason="tool_results_are_archive_evidence_only",
                )
            )
        return candidates

    def _message_candidates(
        self,
        text: str,
        *,
        role: str,
        adapter_id: str,
        session_id: str,
        evidence_id: str,
        project_id: str,
        schema_types: set[MemoryType],
    ) -> list[MemoryCandidateDraft]:
        normalized = self._remember_payload(text)
        candidates: list[MemoryCandidateDraft] = []
        if role in {"assistant", "agent"}:
            if MemoryType.AGENT_EXPERIENCE in schema_types and AGENT_EXPERIENCE_RE.search(text) and EVENT_RE.search(text):
                candidates.append(
                    self._candidate(
                        MemoryType.AGENT_EXPERIENCE,
                        title=self._title(normalized, "Agent experience"),
                        content=normalized,
                        fields={
                            "situation": self._sentence(normalized, 0),
                            "approach": self._sentence(normalized, 1) or normalized[:160],
                            "outcome": self._sentence(normalized, -1) or normalized[-160:],
                            "project_id": project_id,
                            "adapter_id": adapter_id,
                        },
                        confidence=0.8,
                        role=role,
                        adapter_id=adapter_id,
                        session_id=session_id,
                        evidence_id=evidence_id,
                        reason="assistant_final_reusable_experience_hint",
                    )
                )
            return candidates

        if MemoryType.PROJECT_RULE in schema_types and RULE_RE.search(normalized):
            candidates.append(
                self._candidate(
                    MemoryType.PROJECT_RULE,
                    title=self._title(normalized, "Project rule"),
                    content=normalized,
                    fields={"rule": normalized, "project_id": project_id},
                    confidence=0.86 if project_id else 0.68,
                    role=role,
                    adapter_id=adapter_id,
                    session_id=session_id,
                    evidence_id=evidence_id,
                    reason="rule_fallback_hint",
                )
            )
        if MemoryType.PREFERENCE in schema_types and PREFERENCE_RE.search(normalized):
            candidates.append(
                self._candidate(
                    MemoryType.PREFERENCE,
                    title=self._title(normalized, "User preference"),
                    content=normalized,
                    fields={"preference": normalized, "project_id": project_id},
                    confidence=0.82,
                    role=role,
                    adapter_id=adapter_id,
                    session_id=session_id,
                    evidence_id=evidence_id,
                    reason="preference_fallback_hint",
                )
            )
        if MemoryType.PROJECT_DECISION in schema_types and DECISION_RE.search(normalized):
            candidates.append(
                self._candidate(
                    MemoryType.PROJECT_DECISION,
                    title=self._title(normalized, "Project decision"),
                    content=normalized,
                    fields={"decision": normalized, "project_id": project_id},
                    confidence=0.82 if project_id else 0.68,
                    role=role,
                    adapter_id=adapter_id,
                    session_id=session_id,
                    evidence_id=evidence_id,
                    reason="decision_fallback_hint",
                )
            )
        if MemoryType.PROFILE in schema_types and PROFILE_RE.search(normalized):
            candidates.append(
                self._candidate(
                    MemoryType.PROFILE,
                    title=self._title(normalized, "User profile"),
                    content=normalized,
                    fields={"summary": normalized},
                    confidence=0.78,
                    role=role,
                    adapter_id=adapter_id,
                    session_id=session_id,
                    evidence_id=evidence_id,
                    reason="profile_fallback_hint",
                )
            )
        if MemoryType.EVENT in schema_types and EVENT_RE.search(normalized) and not DECISION_RE.search(normalized):
            candidates.append(
                self._candidate(
                    MemoryType.EVENT,
                    title=self._title(normalized, "Event"),
                    content=normalized,
                    fields={"event": normalized, "project_id": project_id},
                    confidence=0.72,
                    role=role,
                    adapter_id=adapter_id,
                    session_id=session_id,
                    evidence_id=evidence_id,
                    reason="event_fallback_hint",
                )
            )
        entity_match = ENTITY_RE.search(normalized)
        if MemoryType.ENTITY in schema_types and entity_match:
            candidates.append(
                self._candidate(
                    MemoryType.ENTITY,
                    title=f"{entity_match.group(1)} {entity_match.group(2)}",
                    content=normalized,
                    fields={"entity_type": entity_match.group(1), "name": entity_match.group(2), "project_id": project_id},
                    confidence=0.78,
                    role=role,
                    adapter_id=adapter_id,
                    session_id=session_id,
                    evidence_id=evidence_id,
                    reason="entity_fallback_hint",
                )
            )
        return candidates

    def _candidate(
        self,
        memory_type: MemoryType,
        *,
        title: str,
        content: str,
        fields: dict[str, Any],
        confidence: float,
        role: str,
        adapter_id: str,
        session_id: str,
        evidence_id: str,
        reason: str,
    ) -> MemoryCandidateDraft:
        merge_key = stable_hash([memory_type.value, fields, content], length=20)
        return MemoryCandidateDraft(
            memory_type=memory_type,
            title=title,
            content=content,
            fields={key: value for key, value in fields.items() if value},
            confidence=confidence,
            source_role=role,
            source_adapter_id=adapter_id,
            source_session_id=session_id,
            source_message_ids=[evidence_id],
            evidence=[{"source": evidence_id, "role": role}],
            merge_key=merge_key,
            reason=reason,
        )

    def _remember_payload(self, text: str) -> str:
        for marker in REMEMBER_MARKERS:
            if marker in text:
                payload = text.split(marker, 1)[1].strip()
                if payload:
                    return payload
        return text.strip()

    def _title(self, text: str, fallback: str) -> str:
        return (text.strip().splitlines()[0][:64] or fallback).strip()

    def _sentence(self, text: str, index: int) -> str:
        sentences = [item.strip() for item in re.split(r"(?<=[.!?。！？])\s+", text) if item.strip()]
        if not sentences:
            return ""
        try:
            return sentences[index]
        except IndexError:
            return ""
