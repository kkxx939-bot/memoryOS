"""记忆系统里的兜底提取器。"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.core.ids import stable_hash
from memoryos.memory.admission import MemoryAdmissionGate
from memoryos.memory.canonical.episode import EvidenceEpisode, SessionArchiveEpisodeAdapter
from memoryos.memory.canonical.evidence import (
    ConstraintPolarity,
    EvidenceSignalKind,
    EvidenceSignalMatcher,
    has_explicit_replacement_cue,
)
from memoryos.memory.canonical.formation import CandidateProposalAdapter
from memoryos.memory.canonical.prefetch import PrefetchedMemory
from memoryos.memory.canonical.proposal import MemorySemanticProposal
from memoryos.memory.extraction.memory_extractor import MemoryExtractorBackend
from memoryos.memory.schema import AdmissionDecision, MemoryCandidateDraft, MemoryType, MemoryTypeSchema
from memoryos.memory.view import adapter_id_from_archive, project_id_from_archive

PROFILE_RE = re.compile(
    r"(?i)(^i am\b|i work\b|i speak\b|my (?:primary |preferred |working )?language\b|"
    r"my active project\b|i am working on\b|我是|我在|负责人|长期从事|主要使用|"
    r"我的(?:主要|常用|工作)?语言|我会说|当前(?:的)?项目|我正在(?:参与|负责|开发))"
)
EVENT_RE = re.compile(r"(?i)(completed|implemented|fixed|released|verified|完成|修复|发布|已验证|已实现)")
AGENT_EXPERIENCE_RE = re.compile(
    r"(?i)(reusable|lesson|pattern|approach|outcome|verified|implemented|fixed|经验|可复用|做法|结果|验证)"
)
ENTITY_RE = re.compile(
    r"(?i)(project|tool|product|organization|person|device|concept|项目|工具|产品|组织|人物|设备|概念)[：:\s]+([\w./@-]+)"
)
REMEMBER_MARKERS = ("记住：", "记住:", "remember:", "Remember:")


class RuleFallbackExtractor(MemoryExtractorBackend):
    """没有模型时只发现可复核候选，不授予 canonical 写权限。"""

    semantic_proposal_backend = True
    llm_semantic_backend = False
    pending_only = True
    is_remote = False

    def __init__(self, signal_matcher: EvidenceSignalMatcher | None = None) -> None:
        self.signal_matcher = signal_matcher or EvidenceSignalMatcher()

    def extract_drafts(
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
            evidence_id = str(
                tool_result.get("event_id")
                or tool_result.get("id")
                or tool_result.get("message_id")
                or f"tool_result:{index}"
            )
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
                    evidence_id=evidence_id,
                    reason="tool_results_are_archive_evidence_only",
                )
            )
        return candidates

    def extract(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
    ) -> list[MemorySemanticProposal]:
        """Return only proposals even when the degraded extractor is called directly."""

        episode = SessionArchiveEpisodeAdapter().adapt(archive)
        return self.extract_with_context(
            archive,
            schemas,
            existing_memories=(),
            episode=episode,
        )

    def extract_with_context(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
        *,
        existing_memories: Sequence[PrefetchedMemory],
        episode: EvidenceEpisode,
    ) -> list[MemorySemanticProposal]:
        """Adapt explicit degraded-mode findings into the one proposal pipeline."""

        candidates = self.extract_drafts(archive, schemas)
        for candidate in candidates:
            if not candidate.fields.get("_replacement_explicit"):
                continue
            identity_key = {
                MemoryType.PROJECT_DECISION: "decision_topic",
                MemoryType.PROJECT_RULE: "rule_topic",
                MemoryType.PROFILE: "attribute_key",
                MemoryType.PREFERENCE: "dimension",
            }.get(candidate.memory_type)
            if not identity_key:
                continue
            identity_value = str(candidate.fields.get(identity_key) or "").casefold()
            matches = [
                item
                for item in existing_memories
                if str(getattr(item, "memory_type", "")) == candidate.memory_type.value
                and str(getattr(item, "state", "")).upper() == "ACTIVE"
                and str(dict(getattr(item, "identity_fields", {}) or {}).get(identity_key) or "").casefold()
                == identity_value
            ]
            if len(matches) != 1:
                # Explicit wording without a unique, same-slot active target is
                # reviewable evidence, not authority to guess a target. Keep
                # the evidenced relation so formation can perform an exact
                # Identity V2 slot lookup; without that lookup transition still
                # fails closed into durable pending.
                candidate.fields["_target_resolution_pending"] = True
                continue
            active = matches[0]
            candidate.fields["_related_memory_ids"] = [str(active.uri)] if getattr(active, "uri", "") else []
            candidate.fields["_related_slot_ids"] = (
                [str(active.slot_id)] if getattr(active, "slot_id", "") else []
            )
            candidate.fields["_related_claim_ids"] = (
                [str(active.claim_id)] if getattr(active, "claim_id", "") else []
            )
        adapter = CandidateProposalAdapter()
        gate = MemoryAdmissionGate()
        project_id = project_id_from_archive(archive)
        adapter_id = adapter_id_from_archive(archive)
        reviewable = [
            candidate
            for candidate in candidates
            if gate.evaluate(
                candidate,
                user_id=archive.user_id,
                project_id=project_id,
                adapter_id=adapter_id,
            ).decision
            in {AdmissionDecision.ACCEPT, AdmissionDecision.PENDING}
        ]
        return [adapter.adapt(candidate, episode, archive) for candidate in reviewable]

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
            if (
                MemoryType.AGENT_EXPERIENCE in schema_types
                and AGENT_EXPERIENCE_RE.search(text)
                and EVENT_RE.search(text)
            ):
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

        constraint_signal = self._constraint_polarity(normalized) is not None and not self._preference_negation(
            normalized
        )
        attributed = self._attributed_statement(normalized)
        attributed_without_resolution = attributed and not self._has_user_resolution(normalized)
        rule_fields = self._rule_fields(normalized, project_id)
        if (
            MemoryType.PROJECT_RULE in schema_types
            and constraint_signal
            and rule_fields.get("rule_topic")
            and not attributed
            and role in {"user", "system"}
        ):
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
            and (
                proposal_kinds
                & {
                    EvidenceSignalKind.CONFIRMATION,
                    EvidenceSignalKind.PROPOSAL,
                    EvidenceSignalKind.EVALUATION,
                }
                or decision_fields.get("_semantic_relation") in {"alternative", "supersedes"}
            )
            and not self._evaluation_without_candidate(normalized)
            and decision_fields.get("decision_topic")
            and not attributed_without_resolution
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
        profile_fields = self._profile_fields(normalized)
        if MemoryType.PROFILE in schema_types and PROFILE_RE.search(normalized) and profile_fields:
            candidates.append(
                self._candidate(
                    MemoryType.PROFILE,
                    title=self._title(normalized, "User profile"),
                    content=normalized,
                    fields={**profile_fields, "summary": normalized},
                    confidence=0.78,
                    role=role,
                    adapter_id=adapter_id,
                    session_id=session_id,
                    evidence_id=evidence_id,
                    reason="profile_fallback_hint",
                )
            )
        if (
            MemoryType.EVENT in schema_types
            and EVENT_RE.search(normalized)
            and not signal_kinds
            & {
                EvidenceSignalKind.CONFIRMATION,
                EvidenceSignalKind.PROPOSAL,
                EvidenceSignalKind.EVALUATION,
            }
        ):
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
        # Commas carry contrast, conditions and exceptions. Splitting on them
        # silently turns a conditional rule into an unconditional one.
        # The same is true when a semicolon separates a base rule from its
        # exception, for example ``forbid Redis; unless used as a cache``.
        if re.search(
            r"(?i)(?:\bunless\b|\bexcept(?:\s+when|\s+for)?\b|\bonly\s+if\b|除非|例外|仅限|只有)",
            text,
        ):
            return [text.strip()]
        clauses = [item.strip() for item in re.split(r"[。！？.!?;；]+", text) if item.strip()]
        return clauses or [text.strip()]

    def _decision_fields(self, text: str, project_id: str) -> dict[str, Any]:
        databases = self._database_values(text)
        database = self._selected_database(text, databases)
        if database:
            fields: dict[str, Any] = {
                "decision_topic": "primary_storage_backend",
                "canonical_value": database,
                "decision": text,
                "project_id": project_id,
            }
            if self._alternative_statement(text):
                fields.update(
                    {
                        "_semantic_speech_act": "proposal",
                        "_semantic_commitment": "exploratory",
                        "_semantic_temporal_scope": "current",
                        "_semantic_relation": "alternative",
                    }
                )
            elif self._replacement_statement(text):
                fields.update(
                    {
                        "_semantic_speech_act": "correction",
                        "_semantic_commitment": "confirmed",
                        "_semantic_temporal_scope": "current",
                        "_semantic_relation": "supersedes",
                        "_replacement_explicit": True,
                    }
                )
            elif self._future_option_statement(text):
                fields.update(
                    {
                        "_semantic_speech_act": (
                            "evaluation_request"
                            if re.search(r"(?i)(?:\bevaluat(?:e|ion)\b|评估)", text)
                            else "proposal"
                        ),
                        "_semantic_commitment": "exploratory",
                        "_semantic_temporal_scope": "future",
                        "_semantic_relation": "alternative",
                    }
                )
            elif self._explicit_current_decision(text):
                fields.update(
                    {
                        "_semantic_speech_act": "confirmation",
                        "_semantic_commitment": "confirmed",
                        "_semantic_temporal_scope": "current",
                        "_semantic_relation": "unrelated",
                    }
                )
            else:
                # A database being available is not evidence that it is the
                # selected primary backend. Keep the extracted value reviewable
                # while failing closed on commitment, time and relation.
                fields.update(
                    {
                        "_semantic_speech_act": "confirmation",
                        "_semantic_commitment": "unknown",
                        "_semantic_temporal_scope": "unknown",
                        "_semantic_relation": "ambiguous",
                    }
                )
            return fields
        if len(databases) > 1 and self._undecided_options(text):
            return {
                "decision_topic": "primary_storage_backend",
                "decision": text,
                "options": databases,
                "project_id": project_id,
                "_semantic_speech_act": "proposal",
                "_semantic_commitment": "unknown",
                "_semantic_temporal_scope": "current",
                "_semantic_relation": "alternative",
            }
        return {"decision": text, "project_id": project_id}

    def _rule_fields(self, text: str, project_id: str) -> dict[str, Any]:
        polarity = self._constraint_polarity(text)
        if self._complex_constraint_statement(text):
            # Multiple competing subjects or opposing constraints are not one
            # atomic rule. Archive the source sentence and let a stronger
            # extractor or human review decompose it instead of guessing.
            return {"rule": text, "project_id": project_id}
        semantic_fields = self._rule_semantic_fields(text)
        if re.search(r"(?i)redis", text) and polarity is not None:
            canonical_value = {
                ConstraintPolarity.REQUIRE: "required",
                ConstraintPolarity.CONDITIONAL_REQUIRE: "required",
                ConstraintPolarity.FORBID: "forbidden",
                ConstraintPolarity.CONDITIONAL_FORBID: "forbidden",
                ConstraintPolarity.ALLOW: "allowed",
                ConstraintPolarity.PREFER: "preferred",
                ConstraintPolarity.DISCOURAGE: "discouraged",
            }[polarity]
            fields: dict[str, Any] = {
                "rule_topic": "redis_usage",
                "canonical_value": canonical_value,
                "polarity": polarity.value,
                "rule": text,
                "project_id": project_id,
                **semantic_fields,
            }
            exception = self._exception(text)
            condition = self._condition(text)
            if polarity in {ConstraintPolarity.CONDITIONAL_FORBID, ConstraintPolarity.CONDITIONAL_REQUIRE}:
                if exception:
                    fields["exception"] = exception
                elif condition:
                    fields["condition"] = condition
                else:
                    fields.update(
                        {
                            "_semantic_commitment": "unknown",
                            "_semantic_temporal_scope": "unknown",
                            "_semantic_relation": "ambiguous",
                        }
                    )
            return fields
        topic = self._rule_topic(text)
        fields = {
            "rule_topic": topic,
            "rule": text,
            "project_id": project_id,
            **semantic_fields,
        }
        if topic and polarity is not None:
            fields["polarity"] = polarity.value
            fields["canonical_value"] = {
                ConstraintPolarity.REQUIRE: "required",
                ConstraintPolarity.CONDITIONAL_REQUIRE: "required",
                ConstraintPolarity.FORBID: "forbidden",
                ConstraintPolarity.CONDITIONAL_FORBID: "forbidden",
                ConstraintPolarity.ALLOW: "allowed",
                ConstraintPolarity.PREFER: "preferred",
                ConstraintPolarity.DISCOURAGE: "discouraged",
            }[polarity]
        return fields

    def _constraint_polarity(self, text: str) -> ConstraintPolarity | None:
        usable = [
            match
            for match in self.signal_matcher.match(text)
            if match.kind == EvidenceSignalKind.CONSTRAINT
            and match.polarity is not None
            and not (match.negated or match.quoted or match.metalinguistic)
        ]
        return usable[-1].polarity if usable else None

    def _exception(self, text: str) -> str:
        preposed = re.search(
            r"(?i)^\s*(?:除非|unless)\s*(.+?)(?:，|,)\s*(?:(?:否则|otherwise)\s*)?",
            text,
        )
        if preposed:
            return preposed.group(1).strip(" ，,。.")
        match = re.search(
            r"(?i)(?:除非|unless|except(?:\s+when|\s+for)?|only\s+if)\s*(.+)$",
            text,
        )
        return match.group(1).strip(" ，,。.") if match else ""

    def _condition(self, text: str) -> str:
        match = re.search(
            r"(?i)^(?:如果|若|当|仅当|只有|if|when|only\s+if)\s*(.+?)[，,]\s*"
            r"(?:必须|需要|要求|不得|禁止|不允许|不要|must|required|require|must\s+not|do\s+not)",
            text,
        )
        return match.group(1).strip(" ，,") if match else ""

    def _rule_semantic_fields(self, text: str) -> dict[str, str]:
        if self._uncertain_rule_statement(text):
            return {
                "_semantic_speech_act": "proposal",
                "_semantic_commitment": "unknown",
                "_semantic_temporal_scope": "unknown",
                "_semantic_relation": "ambiguous",
            }
        if re.search(r"(?i)(?:\bfuture\b|\blater\b|以后|未来|稍后|届时)", text):
            return {
                "_semantic_speech_act": "proposal",
                "_semantic_commitment": "intended",
                "_semantic_temporal_scope": "future",
                "_semantic_relation": "unrelated",
            }
        return {
            "_semantic_speech_act": "confirmation",
            "_semantic_commitment": "confirmed",
            "_semantic_temporal_scope": "current",
            "_semantic_relation": "unrelated",
        }

    def _uncertain_rule_statement(self, text: str) -> bool:
        return bool(
            re.search(
                r"(?i)(?:可能|也许|或许|大概|未确定|不确定|\bmaybe\b|\bmight\b|\bpossibly\b|"
                r"\bmay\s+(?:need|require)\b|\bcould\s+(?:need|require|use)\b)",
                text,
            )
        )

    def _complex_constraint_statement(self, text: str) -> bool:
        usable = [
            match
            for match in self.signal_matcher.match(text)
            if match.kind == EvidenceSignalKind.CONSTRAINT
            and match.polarity is not None
            and not (match.negated or match.quoted or match.metalinguistic)
        ]
        if not usable:
            return False
        subjects = set(self._database_values(text))
        if re.search(r"(?i)\bredis\b", text):
            subjects.add("redis")
        if len(subjects) > 1 and re.search(
            r"(?i)(?:或者|或是|或(?!许)|以及|和|及|、|/|\bor\b|\band\b)", text
        ):
            return True
        if len(usable) < 2:
            return False
        explicit_correction = bool(
            re.search(r"(?i)(?:不是.{0,32}而是|not\s+.{0,32}\bbut\b|rather\s+than)", text)
        )
        if len({match.polarity for match in usable}) > 1 and not explicit_correction:
            return True
        return len(subjects) > 1 and bool(re.search(r"(?i)(?:但是|不过|而是|但|\bbut\b|\bhowever\b)", text))

    def _profile_fields(self, text: str) -> dict[str, Any]:
        project_role = re.search(
            r"(?i)(?:我是|i am)\s*([^，,。.!]+?)\s*(?:的|the)?\s*(负责人|maintainer|owner|lead)", text
        )
        if project_role:
            subject = project_role.group(1).strip()
            role = project_role.group(2).strip()
            return {
                "attribute_key": "project_role",
                "canonical_value": f"{subject}:{role}",
                "active_project": subject,
            }
        active_project = re.search(
            r"(?i)(?:我的?当前(?:的)?项目(?:是|为)|我正在(?:参与|负责|开发)|"
            r"my active project is|i am working on)\s*([^，,。.!]+?)(?:\s*项目|\s+project)?$",
            text,
        )
        if active_project:
            return {"attribute_key": "active_project", "canonical_value": active_project.group(1).strip()}
        location = re.search(r"(?i)(?:我在|i work in|i am based in)\s*([^，,。.!]+?)(?:工作|办公|\bwork\b|$)", text)
        if location:
            return {"attribute_key": "work_location", "canonical_value": location.group(1).strip()}
        occupation = re.search(
            r"(?i)(?:我是|i am an?|i work as an?)\s*([^，,。.!]*(?:工程师|研究员|教授|教师|医生|设计师|经理|"
            r"engineer|researcher|professor|teacher|doctor|designer|manager|developer|tester))$",
            text,
        )
        if occupation:
            return {"attribute_key": "occupation", "canonical_value": occupation.group(1).strip()}
        language = re.search(
            r"(?i)(?:我的(?:主要|常用|工作)?语言(?:是|为)|我会说|"
            r"my (?:primary |preferred |working )?language is|i speak)\s*([^，,。.!]+)",
            text,
        )
        if language:
            return {"attribute_key": "language", "canonical_value": language.group(1).strip()}
        skill = re.search(r"(?i)(?:主要使用|primarily use|mainly use)\s*([^，,。.!]+)", text)
        if skill:
            return {"attribute_key": "primary_skill", "canonical_value": skill.group(1).strip()}
        return {}

    def _rule_topic(self, text: str) -> str:
        patterns = (
            (r"(?i)(source[- ]only|source code|源码).*(audit|审计)|(?:audit|审计).*(source|源码)", "source_audit"),
            (r"(?i)(operationcommitter|write path|写入链路|提交链路)", "canonical_write_path"),
            (r"(?i)(l0|l1|l2|uri tree|uri trees|uri 树|路径树)", "context_layer_uri"),
            (
                r"(?i)(pytest|ruff|test|lint).*(merge|commit|合并|提交)|(?:merge|commit|合并|提交).*(pytest|ruff|test|lint)",
                "pre_merge_verification",
            ),
            (r"(?i)(raw tool output|tool output|原始工具输出)", "raw_tool_output_retention"),
            (r"(?i)(schema metadata|模式元数据|结构化元数据)", "memory_schema_metadata"),
            (r"(?i)(auto(?:matic)? execution|自动执行)", "automatic_execution"),
        )
        return next((topic for pattern, topic in patterns if re.search(pattern, text)), "")

    def _preference_dimension(self, text: str) -> str:
        patterns = (
            (r"(?i)(code review|reviews?|代码审查)", "code_review_style"),
            (
                r"(?i)(concise|findings? first|output|response|answer|final report|简洁|输出|回答|报告)",
                "response_style",
            ),
            (r"(?i)(temperature|degrees?|\d+\s*度|温度)", "temperature"),
            (r"(?i)(air conditioner|air conditioning|direct airflow|空调|直吹)", "climate_comfort"),
            (r"(?i)(room|environment|房间|环境)", "environment_preference"),
            (r"(?i)(sqlite|postgres(?:ql)?|mysql|database|数据库)", "storage_backend"),
        )
        return next((dimension for pattern, dimension in patterns if re.search(pattern, text)), "")

    def _database_value(self, text: str) -> str:
        return self._selected_database(text, self._database_values(text))

    def _database_values(self, text: str) -> list[str]:
        values = []
        for match in re.finditer(r"(?i)\b(sqlite|postgres(?:ql)?|mysql|mariadb|mongodb)\b", text):
            value = match.group(1).casefold()
            normalized = "postgresql" if value == "postgres" else value
            if normalized not in values:
                values.append(normalized)
        return values

    def _selected_database(self, text: str, values: list[str]) -> str:
        if not values:
            return ""
        if len(values) == 1:
            return values[0]
        if self._undecided_options(text):
            return ""
        final = re.search(
            r"(?i)(?:最终(?:决定)?|最后(?:决定)?|finally(?:\s+decided)?|正式(?:改成|改为)|(?:改成|改为|切换到)|(?:switch(?:ed)?\s+to|replace(?:d)?\s+with))(.+)$",
            text,
        )
        if final:
            selected = self._database_values(final.group(1))
            if selected:
                return selected[-1]
        if self._alternative_statement(text):
            backup = re.search(
                r"(?i)\b(sqlite|postgres(?:ql)?|mysql|mariadb|mongodb)\b.{0,20}(?:备用|备选|backup|alternative|option)",
                text,
            )
            if backup:
                value = backup.group(1).casefold()
                return "postgresql" if value == "postgres" else value
        return ""

    def _replacement_statement(self, text: str) -> bool:
        return has_explicit_replacement_cue(text)

    def _explicit_current_decision(self, text: str) -> bool:
        """Return true only for wording that actually selects a current value."""

        return bool(
            re.search(
                r"(?i)(?:最终(?:决定)?|最后(?:决定)?|正式决定|决定(?:使用|采用)|"
                r"继续使用|仍然(?:保持|使用)|保持不变|保持(?:为|使用)|作为当前|确认为当前|当前(?:使用|采用)|"
                r"(?:finally|decided?|adopt(?:ed)?|continue\s+using|remain(?:s|ed)?)(?:\s+to|\s+as|\s+using)?)",
                text,
            )
        )

    def _future_option_statement(self, text: str) -> bool:
        return bool(
            re.search(
                r"(?i)(?:\bfuture\b|\blater\b|\bevaluat(?:e|ion)\b|\bcan\s+consider\b|"
                r"以后|未来|稍后|评估|候选|可以考虑)",
                text,
            )
        )

    def _alternative_statement(self, text: str) -> bool:
        return bool(re.search(r"(?i)(备用|备选|alternative|backup|option|也可以|同时保留)", text))

    def _undecided_options(self, text: str) -> bool:
        return bool(
            re.search(r"(?i)(暂时|尚未|还没|没有).{0,12}(决定|确定)|(?:都可以|either).{0,20}(?:未决定|not decided)", text)
            or (len(self._database_values(text)) > 1 and re.search(r"(?i)(?:\sor\s|或者|或).*(?:都可以|未决定|not decided)", text))
        )

    def _attributed_statement(self, text: str) -> bool:
        return bool(
            re.search(
                r"(?i)(?:他说|她说|他们说|有人(?:说|建议|要求|认为)|"
                r"我(?:曾经?)?听说|听说|据说|传闻|\bi\s+heard\b|\breportedly\b|\baccording\s+to\b|"
                r"(?:[A-Za-z0-9_\u4e00-\u9fff]{1,20})\s*(?:说|表示|声称|认为|建议|提到)|"
                r"he\s+(?:said|suggested|required)|she\s+(?:said|suggested|required)|"
                r"they\s+(?:said|suggested|required)|someone\s+(?:said|suggested|required|recommended)|"
                r"(?:[A-Za-z0-9_]{1,32})\s+(?:said|says|suggested|recommended|claimed))",
                text,
            )
        )

    def _has_user_resolution(self, text: str) -> bool:
        # A later first-person final choice is the user's own proposition, not
        # the quoted speaker's. Other attributed content stays out of fallback.
        return bool(
            re.search(
                r"(?i)(?:但|但是|不过|而我|but).{0,80}(?:我|we)"
                r".{0,32}(?:最终(?:决定)?|最后(?:决定)?|决定(?:使用|采用)|正式(?:改成|改为)|"
                r"decid(?:e|ed)|adopt(?:ed)?)",
                text,
            )
        )

    def _evaluation_without_candidate(self, text: str) -> bool:
        return EvidenceSignalKind.EVALUATION in self._signal_kinds(text) and not self._database_value(text)

    def _signal_kinds(self, text: str, *, allow_hypothetical_proposals: bool = False) -> set[EvidenceSignalKind]:
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
