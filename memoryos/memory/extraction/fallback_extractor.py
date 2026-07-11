"""记忆系统里的兜底提取器。"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.core.ids import stable_hash
from memoryos.memory.canonical.evidence import EvidenceSignalKind, EvidenceSignalMatcher
from memoryos.memory.extraction.memory_extractor import MemoryExtractorBackend
from memoryos.memory.schema import MemoryCandidateDraft, MemoryType, MemoryTypeSchema
from memoryos.memory.view import adapter_id_from_archive, project_id_from_archive

PROFILE_RE = re.compile(r"(?i)(^i am\b|i work\b|我是|我在|负责人|长期从事)")
EVENT_RE = re.compile(r"(?i)(completed|implemented|fixed|released|verified|完成|修复|发布|已验证|已实现)")
AGENT_EXPERIENCE_RE = re.compile(r"(?i)(reusable|lesson|pattern|approach|outcome|verified|implemented|fixed|经验|可复用|做法|结果|验证)")
ENTITY_RE = re.compile(r"(?i)(project|tool|product|organization|person|device|concept|项目|工具|产品|组织|人物|设备|概念)[：:\s]+([\w./@-]+)")
REMEMBER_MARKERS = ("记住：", "记住:", "remember:", "Remember:")


class RuleFallbackExtractor(MemoryExtractorBackend):
    """没有模型时做保守提取，只处理能稳定识别的语义。"""

    candidate_backend = True

    def __init__(self, signal_matcher: EvidenceSignalMatcher | None = None) -> None:
        self.signal_matcher = signal_matcher or EvidenceSignalMatcher()

    def extract(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
    ) -> list[MemoryCandidateDraft]:
        """处理 extract 这一步。"""

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
            clauses = self._clauses(text)
            for clause in clauses:
                candidates.extend(
                    self._message_candidates(
                        clause,
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
        signal_kinds = self._signal_kinds(normalized)
        proposal_kinds = self._signal_kinds(normalized, allow_hypothetical_proposals=True)
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
                            "task_pattern": "reusable_agent_experience",
                            "environment_signature": project_id or adapter_id or "agent",
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

        constraint_signal = EvidenceSignalKind.CONSTRAINT in signal_kinds and not self._preference_negation(normalized)
        rule_fields = self._rule_fields(normalized, project_id)
        if MemoryType.PROJECT_RULE in schema_types and constraint_signal and rule_fields.get("rule_topic"):
            candidates.append(
                self._candidate(
                    MemoryType.PROJECT_RULE,
                    title=self._title(normalized, "Project rule"),
                    content=normalized,
                    fields=rule_fields,
                    confidence=0.86 if project_id else 0.68,
                    role=role,
                    adapter_id=adapter_id,
                    session_id=session_id,
                    evidence_id=evidence_id,
                    reason="rule_fallback_hint",
                )
            )
        preference_dimension = self._preference_dimension(normalized)
        if (
            MemoryType.PREFERENCE in schema_types
            and EvidenceSignalKind.PREFERENCE in signal_kinds
            and preference_dimension
        ):
            candidates.append(
                self._candidate(
                    MemoryType.PREFERENCE,
                    title=self._title(normalized, "User preference"),
                    content=normalized,
                    fields={
                        "subject": "user",
                        "dimension": preference_dimension,
                        "preference": normalized,
                        "project_id": project_id,
                    },
                    confidence=0.82,
                    role=role,
                    adapter_id=adapter_id,
                    session_id=session_id,
                    evidence_id=evidence_id,
                    reason="preference_fallback_hint",
                )
            )
        decision_fields = self._decision_fields(normalized, project_id)
        if (
            MemoryType.PROJECT_DECISION in schema_types
            and proposal_kinds
            & {
                EvidenceSignalKind.CONFIRMATION,
                EvidenceSignalKind.PROPOSAL,
                EvidenceSignalKind.EVALUATION,
            }
            and not self._evaluation_without_candidate(normalized)
            and decision_fields.get("decision_topic")
        ):
            candidates.append(
                self._candidate(
                    MemoryType.PROJECT_DECISION,
                    title=self._title(normalized, "Project decision"),
                    content=normalized,
                    fields=decision_fields,
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
                    fields={"attribute_key": "profile_summary", "summary": normalized},
                    confidence=0.78,
                    role=role,
                    adapter_id=adapter_id,
                    session_id=session_id,
                    evidence_id=evidence_id,
                    reason="profile_fallback_hint",
                )
            )
        if MemoryType.EVENT in schema_types and EVENT_RE.search(normalized) and not signal_kinds & {
            EvidenceSignalKind.CONFIRMATION,
            EvidenceSignalKind.PROPOSAL,
            EvidenceSignalKind.EVALUATION,
        }:
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
                    fields={
                        "entity_type": entity_match.group(1),
                        "canonical_entity_id": entity_match.group(2),
                        "name": entity_match.group(2),
                        "project_id": project_id,
                    },
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
        identity_fields = {
            key: value
            for key, value in fields.items()
            if key
            in {
                "attribute_key",
                "subject",
                "dimension",
                "entity_type",
                "canonical_entity_id",
                "rule_topic",
                "decision_topic",
                "event_key",
                "task_pattern",
                "environment_signature",
                "canonical_value",
            }
        }
        merge_key = stable_hash([memory_type.value, identity_fields or {"evidence_id": evidence_id}], length=20)
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

    def _clauses(self, text: str) -> list[str]:
        clauses = [item.strip() for item in re.split(r"[。！？.!?;；,，]+", text) if item.strip()]
        return clauses or [text.strip()]

    def _decision_fields(self, text: str, project_id: str) -> dict[str, Any]:
        database = self._database_value(text)
        if database:
            return {
                "decision_topic": "primary_storage_backend",
                "canonical_value": database,
                "decision": text,
                "project_id": project_id,
            }
        return {"decision": text, "project_id": project_id}

    def _rule_fields(self, text: str, project_id: str) -> dict[str, Any]:
        if re.search(r"(?i)redis", text) and EvidenceSignalKind.CONSTRAINT in self._signal_kinds(text):
            return {
                "rule_topic": "redis_usage",
                "canonical_value": "forbidden",
                "rule": text,
                "project_id": project_id,
            }
        topic = self._rule_topic(text)
        return {"rule_topic": topic, "rule": text, "project_id": project_id}

    def _rule_topic(self, text: str) -> str:
        patterns = (
            (r"(?i)(source[- ]only|source code|源码).*(audit|审计)|(?:audit|审计).*(source|源码)", "source_audit"),
            (r"(?i)(operationcommitter|write path|写入链路|提交链路)", "canonical_write_path"),
            (r"(?i)(l0|l1|l2|uri tree|uri trees|uri 树|路径树)", "context_layer_uri"),
            (r"(?i)(pytest|ruff|test|lint).*(merge|commit|合并|提交)|(?:merge|commit|合并|提交).*(pytest|ruff|test|lint)", "pre_merge_verification"),
            (r"(?i)(raw tool output|tool output|原始工具输出)", "raw_tool_output_retention"),
            (r"(?i)(schema metadata|模式元数据|结构化元数据)", "memory_schema_metadata"),
            (r"(?i)(auto(?:matic)? execution|自动执行)", "automatic_execution"),
        )
        return next((topic for pattern, topic in patterns if re.search(pattern, text)), "")

    def _preference_dimension(self, text: str) -> str:
        patterns = (
            (r"(?i)(code review|reviews?|代码审查)", "code_review_style"),
            (r"(?i)(concise|findings? first|output|response|answer|final report|简洁|输出|回答|报告)", "response_style"),
            (r"(?i)(temperature|degrees?|\d+\s*度|温度)", "temperature"),
            (r"(?i)(air conditioner|air conditioning|direct airflow|空调|直吹)", "climate_comfort"),
            (r"(?i)(room|environment|房间|环境)", "environment_preference"),
            (r"(?i)(sqlite|postgres(?:ql)?|mysql|database|数据库)", "storage_backend"),
        )
        return next((dimension for pattern, dimension in patterns if re.search(pattern, text)), "")

    def _database_value(self, text: str) -> str:
        match = re.search(r"(?i)\b(sqlite|postgres(?:ql)?|mysql|mariadb|mongodb)\b", text)
        if match is None:
            return ""
        value = match.group(1).casefold()
        return "postgresql" if value == "postgres" else value

    def _evaluation_without_candidate(self, text: str) -> bool:
        return EvidenceSignalKind.EVALUATION in self._signal_kinds(text) and not self._database_value(text)

    def _signal_kinds(
        self, text: str, *, allow_hypothetical_proposals: bool = False
    ) -> set[EvidenceSignalKind]:
        return {
            match.kind
            for match in self.signal_matcher.match(text)
            if not (
                match.negated
                or match.quoted
                or match.metalinguistic
                or (
                    match.hypothetical
                    and not (
                        allow_hypothetical_proposals
                        and match.kind in {EvidenceSignalKind.PROPOSAL, EvidenceSignalKind.EVALUATION}
                    )
                )
            )
        }

    def _preference_negation(self, text: str) -> bool:
        return bool(re.search(r"(?i)\bdo\s+not\s+like\b|不喜欢", text))
