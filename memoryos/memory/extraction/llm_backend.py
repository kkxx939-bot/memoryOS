"""记忆系统里的大模型后端。"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.core.ids import stable_hash
from memoryos.memory.canonical.episode import EvidenceEpisode, SessionArchiveEpisodeAdapter
from memoryos.memory.canonical.evidence import EvidenceRef
from memoryos.memory.canonical.prefetch import PrefetchedMemory
from memoryos.memory.canonical.proposal import (
    Atomicity,
    Attribution,
    Commitment,
    Durability,
    EpistemicStatus,
    MemorySemanticProposal,
    ModalForce,
    SemanticAssessment,
    SemanticRelation,
    SpeechAct,
    TemporalScope,
    UtteranceMode,
)
from memoryos.memory.canonical.scope import ScopeRef
from memoryos.memory.extraction.egress import EgressDecision, MemoryEgressPolicy
from memoryos.memory.extraction.errors import (
    MemoryExtractionCandidateValidationError,
    MemoryExtractionConfigurationError,
    MemoryExtractionMalformedEnvelopeError,
    MemoryExtractionSecurityError,
    classify_memory_extraction_failure,
)
from memoryos.memory.schema import MemoryTypeSchema
from memoryos.memory.view import adapter_id_from_archive


class MemoryModelProvider(Protocol):
    """约定 MemoryModelProvider 需要提供的接口。"""

    is_remote: bool

    def complete(self, prompt: str) -> str: ...


_FIELD_MEANINGS: Mapping[str, str] = {
    "attribute_key": "Stable profile attribute category such as occupation, work_location, or project_role.",
    "canonical_value": "Normalized semantic value of the claim, independent of source wording.",
    "summary": "Human-readable display summary; it does not define claim identity.",
    "scope": "Semantic applicability descriptor only; it never grants storage scope authority.",
    "stability": "Expected durability of the profile fact.",
    "subject": "Stable semantic subject of a preference or rule.",
    "dimension": "Stable axis of preference for the subject.",
    "preference": "Human-readable durable preference statement.",
    "preference_value": "Normalized preference value used when canonical_value is insufficient.",
    "value": "Normalized semantic value alias declared by the schema.",
    "project_id": "Project applicability label; it cannot select a storage target or authority boundary.",
    "workspace_id": "Workspace applicability label; it cannot select a storage target or authority boundary.",
    "applies_to": "Explicit applicability qualifier for the semantic claim.",
    "visibility": "Semantic visibility qualifier; it cannot set the system visibility policy.",
    "entity_type": "Stable category of an entity, such as project, tool, product, organization, or person.",
    "canonical_entity_id": "Stable normalized entity identifier, not a database URI.",
    "name": "Canonical human-readable entity name.",
    "aliases": "Alternative human-readable names for the same entity.",
    "event_key": "Stable semantic identity of one atomic event.",
    "event": "Human-readable description of one atomic event.",
    "occurred_at": "Evidence-supported time when the event occurred.",
    "outcome": "Observed or confirmed semantic outcome.",
    "rule_topic": "Stable topic governed by a project rule.",
    "rule": "Human-readable project rule statement.",
    "rationale": "Explanatory display detail; it does not define core claim identity.",
    "decision_topic": "Stable project decision slot whose adopted value may change.",
    "decision": "Human-readable project decision statement.",
    "alternatives": "Options considered but not adopted as the current decision.",
    "decided_at": "Evidence-supported time at which the decision became effective.",
    "task_pattern": "Stable reusable task or situation pattern for agent experience.",
    "environment_signature": "Stable applicability signature of the execution environment.",
    "situation": "Evidence-backed situation in which the experience applies.",
    "approach": "Reusable approach taken in the situation.",
    "adapter_id": "Source adapter label for provenance, not a target selector.",
    "tooling": "Tools relevant to the reusable experience.",
    "environment": "Environment applicability qualifier, such as production or testing.",
    "device": "Device applicability qualifier.",
    "activity": "Activity applicability qualifier.",
    "valid_time": "Time interval in which the claim applies.",
    "condition": "Condition under which the claim applies; it must be preserved structurally.",
    "conditions": "All conditions under which the claim applies; they must be preserved structurally.",
    "exception": "Explicit exception to the base claim; it must never be dropped.",
    "exceptions": "All explicit exceptions to the base claim; they must never be dropped.",
    "applicability_qualifier": "Additional evidence-backed qualifier that changes claim applicability.",
    "title": "Display title; it never defines claim identity.",
    "display_name": "Display name; it never defines claim identity.",
    "display_text": "Display wording; it never defines claim identity.",
    "source_text": "Original source wording retained only for display or evidence.",
    "details": "Explanatory display details; they never define core claim identity.",
    "reason": "Explanatory reason; it never defines core claim identity.",
    "asserted_by": "System-derived principal that asserted the claim; models must not author it.",
    "author": "System-derived source author provenance; models must not author it.",
    "source_role": "Evidence-derived actor role; models may report it only through the dedicated checked field.",
    "source_adapter_id": "System-derived source adapter provenance; models must not author it.",
    "source_session_id": "System-derived source session provenance; models must not author it.",
    "evidence_source": "System-derived evidence source provenance; models must not author it.",
    "extractor_version": "System-owned extractor version used for audit and replay.",
    "model_id": "System-owned model identifier used for audit and replay.",
    "prompt_version": "System-owned prompt contract version used for audit and replay.",
    "identity": "Legacy projected identity payload; canonical identity_fields is the only model-facing identity surface.",
    "evidence": "Legacy projected evidence payload; models must use evidence_refs and field_evidence_refs instead.",
    "topic": "Legacy preference topic projection; the canonical slot uses subject and dimension.",
    "content": "Legacy human-readable content projection; it is derived from canonical semantic fields.",
    "type": "Legacy entity type projection; canonical entity_type is used instead.",
    "date": "Legacy event date projection; evidence-supported occurred_at is used instead.",
    "rule_key": "Legacy rule identity projection; canonical rule_topic is used instead.",
    "constraints": "Legacy projected rule constraints derived from validated canonical rule fields.",
    "decision_key": "Legacy decision identity projection; canonical decision_topic is used instead.",
    "status": "System-derived lifecycle or projection status; models must not author it.",
    "reflect": "Legacy reflection projection derived from validated agent experience fields.",
    "situation_key": "Legacy experience identity projection; canonical task_pattern and environment_signature are used instead.",
}


def _schema_value_fields(schema: MemoryTypeSchema) -> set[str]:
    """Return the finite set of model-authored value fields for one schema."""

    fields = {
        *schema.required_fields,
        *schema.optional_fields,
        *schema.claim_identity_fields,
        *schema.display_fields,
        *schema.applicability_fields,
        "canonical_value",
    }
    fields.discard("*")
    return fields


def _existing_candidate_ref(index: int) -> str:
    return f"existing_{index}"


def _schema_field_contract(schema: MemoryTypeSchema) -> dict[str, dict[str, object]]:
    value_fields = _schema_value_fields(schema)
    field_names = {
        *schema.required_fields,
        *schema.optional_fields,
        *schema.slot_identity_fields,
        *schema.claim_identity_fields,
        *schema.applicability_fields,
        *schema.display_fields,
        *schema.provenance_fields,
        *schema.field_merge_rules,
        "canonical_value",
    }
    field_names.discard("*")
    result: dict[str, dict[str, object]] = {}
    for field_name in sorted(field_names):
        model_authored = field_name in schema.slot_identity_fields or field_name in value_fields
        if field_name in schema.slot_identity_fields:
            identity_role = "slot_identity"
            placement = "identity_fields"
        elif field_name in schema.claim_identity_fields:
            identity_role = "claim_identity"
            placement = "value_fields"
        elif field_name in schema.provenance_fields:
            identity_role = "provenance_only"
            placement = "system_metadata"
        elif field_name in schema.display_fields:
            identity_role = "display_only"
            placement = "value_fields"
        elif field_name in schema.applicability_fields:
            identity_role = "applicability_only"
            placement = "value_fields"
        elif field_name in schema.field_merge_rules and not model_authored:
            identity_role = "legacy_projection_only"
            placement = "legacy_projection"
        else:
            identity_role = "semantic_non_identity"
            placement = "value_fields"
        result[field_name] = {
            "meaning": _FIELD_MEANINGS.get(
                field_name,
                f"Schema-defined {field_name} semantic field; never use it as a storage control.",
            ),
            "placement": placement,
            "identity_role": identity_role,
            "required": model_authored
            and (
                field_name in schema.slot_identity_fields
                or (field_name == "canonical_value" and field_name in schema.claim_identity_fields)
            ),
            "legacy_projection_required": field_name in schema.required_fields,
            "model_authored": model_authored,
        }
    return result


class MemoryExtractionPromptBuilder:
    """负责 MemoryExtractionPromptBuilder 这部分逻辑。"""

    def build(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
        *,
        existing_memories: Sequence[PrefetchedMemory] = (),
        episode: EvidenceEpisode | None = None,
    ) -> str:
        """根据输入组装结果对象。"""

        schema_names = ", ".join(schema.memory_type.value for schema in schemas)
        schema_payload = [
            {
                "memory_type": schema.memory_type.value,
                "legacy_projection_required_fields": list(schema.required_fields),
                "legacy_projection_optional_fields": list(schema.optional_fields),
                "description": schema.description,
                "slot_identity_fields": list(schema.slot_identity_fields),
                "claim_identity_fields": list(schema.claim_identity_fields),
                "field_semantics": _schema_field_contract(schema),
            }
            for schema in schemas
        ]
        existing = json.dumps(
            [
                {
                    "candidate_ref": _existing_candidate_ref(index),
                    "memory_type": item.memory_type,
                    "state": item.state,
                    "canonical_value": item.canonical_value,
                    "identity_fields": item.identity_fields,
                    "scope": item.scope,
                    "l0": item.l0,
                    "l1": item.l1,
                    "l2": item.l2,
                }
                for index, item in enumerate(existing_memories)
            ],
            ensure_ascii=False,
            sort_keys=True,
        )
        episode = episode or SessionArchiveEpisodeAdapter().adapt(archive)
        events = [
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "actor": event.actor.to_dict(),
                "subjects": [subject.to_dict() for subject in event.subjects],
                "event_digest": event.digest,
                "content_hash": EvidenceRef.from_event(event).content_hash,
                "content_path": event.content_path,
                "occurred_at": event.occurred_at.isoformat(),
                "ingested_at": (event.ingested_at or event.occurred_at).isoformat(),
                "sequence": event.sequence,
                "text": event.text(),
            }
            for event in episode.events
        ]
        legal_scopes = [scope.to_dict() for scope in episode.legal_scope_candidates()]
        return (
            "Extract durable memory semantic proposals as JSON using semantic contract v3. "
            "A durable memory is an evidence-backed fact, preference, rule, decision, entity, event, or reusable "
            "experience that remains useful beyond this turn; transient discussion, prediction, raw tool output, "
            "quotation, hypothesis, and unchosen alternatives are not current authoritative memory. "
            "Return an object with a candidates array. Each proposal may describe semantics but is not a database operation. "
            f"Allowed memory_type values: {schema_names}. "
            f"Allowed speech_act values: {', '.join(item.value for item in SpeechAct)}. "
            f"Allowed commitment values: {', '.join(item.value for item in Commitment)}. "
            f"Allowed temporal_scope values: {', '.join(item.value for item in TemporalScope)}. "
            f"Allowed relation_to_existing values: {', '.join(item.value for item in SemanticRelation)}. "
            f"Allowed epistemic_status values: {', '.join(item.value for item in EpistemicStatus)}. "
            f"Allowed utterance_mode values: {', '.join(item.value for item in UtteranceMode)}. "
            f"Allowed attribution values: {', '.join(item.value for item in Attribution)}. "
            f"Allowed durability values: {', '.join(item.value for item in Durability)}. "
            f"Allowed modal_force values: {', '.join(item.value for item in ModalForce)}. "
            f"Allowed atomicity values: {', '.join(item.value for item in Atomicity)}. "
            "Use identity_fields and value_fields separately. Include speech_act, commitment, temporal_scope, "
            "relation_to_existing, utterance_mode, attribution, durability, modal_force, atomicity, "
            "epistemic_status, evidence_refs, atomic_evidence_ref, and field_evidence_refs. "
            "MEMORY_SCHEMAS field_semantics is the authoritative per-type field contract: obey each field's "
            "meaning, placement, and identity_role, and never invent an identity or value field name. "
            "Never output a schema field whose field_semantics model_authored value is false inside identity_fields "
            "or value_fields; provenance, legacy projection, lifecycle, and system metadata are supplied only by "
            "trusted system code. The dedicated top-level source_role report is allowed but is verified against evidence. "
            "A slot identifies the stable subject/topic whose authoritative value may change. A claim identifies one "
            "normalized semantic value plus its applicability conditions. Source wording, display text, rationale, "
            "confidence, and evidence text never belong in claim identity. Put the source-grounded semantic value in "
            "canonical_value; preserve the value wording present in the selected evidence span and let the system apply "
            "registered alias normalization. Never translate or substitute a value that is absent from that span. "
            "Each candidate must contain exactly one atomic fact. Preserve conditions and exceptions structurally. "
            "Set atomicity=ATOMIC only when the candidate is one proposition; use COMPOUND or UNKNOWN otherwise. "
            "utterance_mode distinguishes direct ASSERTION or DIRECTIVE from QUESTION, HYPOTHETICAL, and MIXED text. "
            "attribution=SOURCE_ACTOR only when the event actor directly owns the proposition; reported third-party "
            "claims, quotations, and mixed attribution must remain distinct. durability=DURABLE only when the memory "
            "remains useful beyond this turn. modal_force carries normative direction and is NONE for non-normative facts. "
            "Distinguish the speaker's final decision from quoted advice, hypothetical discussion, and alternatives. "
            "CONFIRMATION only confirms the candidate; it never means SUPERSEDES by itself. ALTERNATIVE and SUPPLEMENTS "
            "must not replace an ACTIVE claim. CORRECTS or SUPERSEDES requires explicit replacement wording, a unique "
            "related active claim, field evidence for that relation, current compatible applicability, and authoritative "
            "source evidence. These destructive relation labels describe semantics only and never authorize an effect; "
            "the system requires a separate structured review before changing or retracting an ACTIVE claim. "
            "Use UNKNOWN or AMBIGUOUS so the system persists PENDING whenever identity, commitment, "
            "temporality, relation, scope authority, condition, exception, or speaker authority is unclear. "
            "User/system assertions can be authoritative; assistant/tool suggestions are not user decisions. "
            "Examples: 'Confirm MySQL can be a backup' => ALTERNATIVE, not SUPERSEDES. "
            "'PostgreSQL stays unchanged; MySQL is a backup' => ALTERNATIVE and do not target PostgreSQL. "
            "'Previously SQLite, now switch to PostgreSQL' => SUPERSEDES with the SQLite active claim as related target. "
            "'Someone suggested MySQL' => proposal or pending, not a user decision. "
            "'Maybe use Redis later' => FUTURE/PROPOSAL, not CURRENT/CONFIRMED. "
            "'Do not use Redis unless it is only a short-term cache' => CONDITIONAL_FORBID Redis with the cache exception retained. "
            "Bind every identity field and value field to a span that literally contains its source value. Bind "
            "semantic.speech_act, semantic.commitment, "
            "semantic.temporal_scope, semantic.relation_to_existing, semantic.utterance_mode, "
            "semantic.attribution, semantic.durability, semantic.modal_force, semantic.atomicity, and transition "
            "to evidence in field_evidence_refs. Use only suggested_scope_refs selected from legal scopes. "
            "atomic_evidence_ref must select one exact non-empty span (event_id, span_start, span_end) from "
            "EPISODE_EVENTS. Bind every semantic.* field and transition to that exact atomic span. Other evidence_refs "
            "may supply read-only conversational context but cannot supply the source actor's authority. "
            "Use related_candidate_refs only with opaque candidate_ref values present in EXISTING_MEMORIES. "
            "The system maps those opaque refs to internal identities after validation. Never output a URI, slot id, "
            "claim id, operation, or storage target. Do not output tenant IDs, visibility policy, revisions, DELETE, or scope moves.\n"
            f"LEGAL_SCOPES={json.dumps(legal_scopes, ensure_ascii=False, sort_keys=True)}\n"
            f"MEMORY_SCHEMAS={json.dumps(schema_payload, ensure_ascii=False, sort_keys=True)}\n"
            f"EXISTING_MEMORIES={existing}\n"
            f"EPISODE_EVENTS={json.dumps(events, ensure_ascii=False, sort_keys=True)}"
        )


class _MemoryExtractionJsonParser:
    """只负责拆出并解析严格的 JSON 响应信封。"""

    def _load_json(self, response: str) -> dict | list:
        text = response.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MemoryExtractionMalformedEnvelopeError(f"LLM memory response is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict | list):
            raise MemoryExtractionMalformedEnvelopeError("LLM memory response must be an object or list")
        return payload


@dataclass(frozen=True)
class RejectedMemoryCandidate:
    index: int
    proposal_id: str
    reason: str
    security_flags: tuple[str, ...] = ()
    security_error: bool = False


@dataclass(frozen=True)
class MemoryExtractionBatchResult:
    accepted: tuple[MemorySemanticProposal, ...]
    rejected: tuple[RejectedMemoryCandidate, ...] = ()
    security_flags: tuple[str, ...] = ()
    egress_decision: str = EgressDecision.LOCAL_ONLY.value
    egress_audit: dict[str, str] | None = None


class LLMMemoryExtractorBackend:
    """让模型只产出语义提案，不能直接决定增删改。"""

    semantic_proposal_backend = True
    llm_semantic_backend = True
    egress_policy_enforced = True
    handles_retries = True
    _PROPOSAL_FIELDS = {
        "proposal_id",
        "memory_type",
        "identity_fields",
        "value_fields",
        "semantic",
        "epistemic_status",
        "suggested_scope_refs",
        "related_candidate_refs",
        "evidence_refs",
        "atomic_evidence_ref",
        "field_evidence_refs",
        "confidence",
        "source_role",
    }
    _SEMANTIC_FIELDS = {
        "speech_act",
        "commitment",
        "temporal_scope",
        "relation_to_existing",
        "utterance_mode",
        "attribution",
        "durability",
        "modal_force",
        "atomicity",
    }
    _SCOPE_FIELDS = {
        "namespace",
        "kind",
        "id",
        "parent_id",
        "attributes",
        "parent_path",
        "confidence",
        "source",
        "inferred",
    }
    _SPEECH_VALUES = {item.value.casefold() for item in SpeechAct}
    _COMMITMENT_VALUES = {item.value.casefold() for item in Commitment}
    _TEMPORAL_VALUES = {item.value.casefold() for item in TemporalScope}
    _RELATION_VALUES = {item.value.casefold() for item in SemanticRelation}
    _UTTERANCE_VALUES = {item.value.casefold() for item in UtteranceMode}
    _ATTRIBUTION_VALUES = {item.value.casefold() for item in Attribution}
    _DURABILITY_VALUES = {item.value.casefold() for item in Durability}
    _MODAL_FORCE_VALUES = {item.value.casefold() for item in ModalForce}
    _ATOMICITY_VALUES = {item.value.casefold() for item in Atomicity}
    _FORBIDDEN_CONTROL_FIELDS = {
        "action",
        "actions",
        "delete",
        "deletion",
        "operation",
        "operations",
        "target",
        "targets",
        "tenant_id",
        "owner_user_id",
        "user_id",
        "revision",
        "expected_revision",
        "visibility_policy",
        "scope_move",
        "update",
        "updates",
    }

    def __init__(
        self,
        provider: MemoryModelProvider,
        prompt_builder: MemoryExtractionPromptBuilder | None = None,
        parser: _MemoryExtractionJsonParser | None = None,
        model_id: str | None = None,
        extractor_version: str = "llm_semantic_extractor_v3",
        egress_policy: MemoryEgressPolicy | None = None,
        max_attempts: int = 3,
        retry_backoff_seconds: Sequence[float] = (0.0, 0.05, 0.2),
    ) -> None:
        if not callable(getattr(provider, "complete", None)):
            raise MemoryExtractionConfigurationError("memory model provider has no complete() method")
        if max_attempts < 1:
            raise MemoryExtractionConfigurationError("max_attempts must be positive")
        self.provider = provider
        self.prompt_builder = prompt_builder or MemoryExtractionPromptBuilder()
        self.parser = parser or _MemoryExtractionJsonParser()
        self.model_id = model_id
        self.extractor_version = extractor_version
        self.prompt_version = "memory_semantic_proposal_v3"
        self.semantic_contract_version = "v3"
        self.egress_policy = egress_policy or MemoryEgressPolicy()
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = tuple(float(item) for item in retry_backoff_seconds)
        self.last_egress_audit: dict[str, str] | None = None

    def extract(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
    ) -> list[MemorySemanticProposal]:
        """处理 extract 这一步。"""

        episode = SessionArchiveEpisodeAdapter().adapt(archive)
        return list(self.extract_with_context(archive, schemas, existing_memories=(), episode=episode))

    def extract_with_context(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
        *,
        existing_memories: Sequence[PrefetchedMemory],
        episode: EvidenceEpisode,
    ) -> list[MemorySemanticProposal]:
        """Compatibility surface returning only accepted proposals."""

        result = self.extract_batch_with_context(
            archive,
            schemas,
            existing_memories=existing_memories,
            episode=episode,
        )
        return list(result.accepted)

    def extract_batch_with_context(
        self,
        archive: SessionArchive,
        schemas: Sequence[MemoryTypeSchema],
        *,
        existing_memories: Sequence[PrefetchedMemory],
        episode: EvidenceEpisode,
    ) -> MemoryExtractionBatchResult:
        """Parse a trusted envelope while isolating each candidate failure."""

        remote = self._provider_is_remote()
        assessment = self.egress_policy.evaluate(
            archive,
            episode,
            remote=remote,
            existing_memories=existing_memories,
        )
        if remote and assessment.decision in {EgressDecision.DENY, EgressDecision.LOCAL_ONLY}:
            denied_flags = (
                ("privacy_egress_blocked",)
                if any(item.value in {"secret", "restricted_scope"} for item in assessment.categories)
                else ("egress_denied",)
            )
            self.last_egress_audit = {
                "outbound_digest": "",
                "decision": assessment.decision.value,
                "provider": type(self.provider).__name__,
                "model": str(self.model_id or ""),
            }
            return MemoryExtractionBatchResult(
                (),
                (
                    RejectedMemoryCandidate(
                        index=-1,
                        proposal_id="",
                        reason="remote_egress_policy_blocked_sensitive_archive",
                        security_flags=denied_flags,
                        security_error=True,
                    ),
                ),
                denied_flags,
                assessment.decision.value,
                dict(self.last_egress_audit),
            )

        prompt = self.prompt_builder.build(
            archive,
            schemas,
            existing_memories=existing_memories,
            episode=episode,
        )
        prompt = self.egress_policy.redact(prompt, assessment)
        self.last_egress_audit = {
            "outbound_digest": stable_hash([prompt], length=64),
            "decision": assessment.decision.value,
            "provider": type(self.provider).__name__,
            "model": str(self.model_id or ""),
        }
        payload = self._complete_and_parse(prompt)
        if not isinstance(payload, dict):
            raise MemoryExtractionMalformedEnvelopeError(
                "semantic memory extraction response must be an object with candidates"
            )
        if set(payload) != {"candidates"}:
            unknown = set(payload) - {"candidates"}
            raise MemoryExtractionMalformedEnvelopeError(
                f"memory extraction response contains unknown fields: {','.join(sorted(unknown))}"
            )
        raw_candidates = payload.get("candidates", [])
        if not isinstance(raw_candidates, list):
            raise MemoryExtractionMalformedEnvelopeError("memory extraction candidates must be a list")
        allowed = {schema.memory_type.value for schema in schemas}
        schemas_by_type = {schema.memory_type.value: schema for schema in schemas}
        proposals: list[MemorySemanticProposal] = []
        rejected: list[RejectedMemoryCandidate] = []
        proposal_ids: set[str] = set()
        legal_scopes = {scope.key: scope for scope in episode.legal_scope_candidates()}
        existing_candidates = {_existing_candidate_ref(index): item for index, item in enumerate(existing_memories)}
        for index, raw in enumerate(raw_candidates):
            raw_id = str(raw.get("proposal_id") or "") if isinstance(raw, dict) else ""
            try:
                proposal = self._proposal_from_raw(
                    index,
                    raw,
                    archive=archive,
                    episode=episode,
                    allowed=allowed,
                    schemas_by_type=schemas_by_type,
                    legal_scopes=legal_scopes,
                    existing_candidates=existing_candidates,
                    proposal_ids=proposal_ids,
                )
            except (KeyError, TypeError, ValueError) as exc:
                reason = str(exc)
                flags = self._rejection_flags(reason)
                security_flags = {
                    "evidence_integrity_rejected",
                    "source_authority_rejected",
                    "scope_authority_rejected",
                    "target_reference_rejected",
                    "forbidden_output_field_rejected",
                    "operation_control_rejected",
                    "revision_control_rejected",
                }
                typed_error: MemoryExtractionCandidateValidationError | MemoryExtractionSecurityError
                if security_flags.intersection(flags):
                    typed_error = MemoryExtractionSecurityError(reason)
                else:
                    typed_error = MemoryExtractionCandidateValidationError(reason)
                rejected.append(
                    RejectedMemoryCandidate(
                        index=index,
                        proposal_id=raw_id,
                        reason=str(typed_error),
                        security_flags=flags,
                        security_error=isinstance(typed_error, MemoryExtractionSecurityError),
                    )
                )
                continue
            proposal_ids.add(proposal.proposal_id)
            proposals.append(proposal)
        flags = tuple(dict.fromkeys(flag for item in rejected for flag in item.security_flags))
        return MemoryExtractionBatchResult(
            tuple(proposals),
            tuple(rejected),
            flags,
            assessment.decision.value,
            dict(self.last_egress_audit),
        )

    def _complete_and_parse(self, prompt: str) -> dict | list:
        """Retry only failures whose typed contract explicitly permits retry."""

        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                response = self.provider.complete(prompt)
                if not isinstance(response, str):
                    raise MemoryExtractionMalformedEnvelopeError("memory model provider response must be text")
                payload = self.parser._load_json(response)
                if not isinstance(payload, dict):
                    raise MemoryExtractionMalformedEnvelopeError(
                        "semantic memory extraction response must be an object with candidates"
                    )
                if set(payload) != {"candidates"}:
                    unknown = set(payload) - {"candidates"}
                    raise MemoryExtractionMalformedEnvelopeError(
                        "memory extraction response contains unknown fields: " + ",".join(sorted(unknown))
                    )
                if not isinstance(payload.get("candidates"), list):
                    raise MemoryExtractionMalformedEnvelopeError("memory extraction candidates must be a list")
                return payload
            except MemoryExtractionMalformedEnvelopeError as exc:
                last_error = exc
            except Exception as exc:
                last_error = classify_memory_extraction_failure(exc)
                if isinstance(last_error, MemoryExtractionConfigurationError):
                    raise last_error from exc
            if attempt + 1 >= self.max_attempts:
                assert last_error is not None
                raise last_error
            delay = (
                self.retry_backoff_seconds[min(attempt, len(self.retry_backoff_seconds) - 1)]
                if self.retry_backoff_seconds
                else 0.0
            )
            if delay > 0:
                time.sleep(delay)
        raise AssertionError("unreachable extraction retry state")

    def _provider_is_remote(self) -> bool:
        """Unknown providers are remote by default; local providers must opt in explicitly."""

        return getattr(self.provider, "is_remote", True) is not False

    @property
    def is_remote(self) -> bool:
        return self._provider_is_remote()

    def _proposal_from_raw(
        self,
        index: int,
        raw: Any,
        *,
        archive: SessionArchive,
        episode: EvidenceEpisode,
        allowed: set[str],
        schemas_by_type: dict[str, MemoryTypeSchema],
        legal_scopes: Mapping[str, ScopeRef],
        existing_candidates: Mapping[str, PrefetchedMemory],
        proposal_ids: set[str],
    ) -> MemorySemanticProposal:
        if not isinstance(raw, dict):
            raise ValueError(f"candidate[{index}] must be an object")
        unknown = set(raw) - self._PROPOSAL_FIELDS
        if unknown:
            raise ValueError(f"candidate[{index}] contains unknown fields: {','.join(sorted(unknown))}")
        forbidden_paths = self._forbidden_nested_control_paths(raw)
        if forbidden_paths:
            raise ValueError(f"candidate[{index}] contains forbidden control fields: {','.join(forbidden_paths)}")
        memory_type = str(raw.get("memory_type", ""))
        if memory_type not in allowed:
            raise ValueError(f"candidate[{index}] memory_type is not allowed: {memory_type}")
        identity_fields = raw.get("identity_fields", {})
        value_fields = raw.get("value_fields", {})
        semantic = raw.get("semantic", {})
        if (
            not isinstance(identity_fields, dict)
            or not isinstance(value_fields, dict)
            or not isinstance(semantic, dict)
        ):
            raise ValueError(f"candidate[{index}] semantic fields must be objects")
        if (
            not identity_fields
            or not value_fields
            or not any(value is not None and value != "" for value in value_fields.values())
        ):
            raise ValueError(f"candidate[{index}] requires non-empty identity_fields and value_fields")
        schema = schemas_by_type[memory_type]
        expected_identity = set(schema.slot_identity_fields)
        missing_identity = {
            field_name
            for field_name in expected_identity
            if identity_fields.get(field_name) is None or identity_fields.get(field_name) == ""
        }
        unknown_identity = set(identity_fields) - expected_identity
        if missing_identity or unknown_identity:
            details = [
                *(f"missing:{field_name}" for field_name in sorted(missing_identity)),
                *(f"unknown:{field_name}" for field_name in sorted(unknown_identity)),
            ]
            raise ValueError(f"candidate[{index}] slot identity mismatch: {','.join(details)}")
        unknown_value_fields = set(value_fields) - _schema_value_fields(schema)
        if unknown_value_fields:
            raise ValueError(
                f"candidate[{index}] value_fields contains fields outside the {memory_type} schema: "
                f"{','.join(sorted(unknown_value_fields))}"
            )
        if "canonical_value" in schema.claim_identity_fields and (
            value_fields.get("canonical_value") is None or value_fields.get("canonical_value") == ""
        ):
            raise ValueError(f"candidate[{index}] requires value_fields.canonical_value")
        confidence = self._validated_candidate_confidence(index, raw.get("confidence", 0.5))
        semantic_fields = set(semantic)
        if semantic_fields != self._SEMANTIC_FIELDS:
            missing_semantic = self._SEMANTIC_FIELDS - semantic_fields
            unknown_semantic = semantic_fields - self._SEMANTIC_FIELDS
            details = [
                *(f"missing:{field_name}" for field_name in sorted(missing_semantic)),
                *(f"unknown:{field_name}" for field_name in sorted(unknown_semantic)),
            ]
            raise ValueError(f"candidate[{index}] semantic fields mismatch: {','.join(details)}")
        self._validate_semantic_enums(index, semantic)
        evidence_refs = self._evidence_refs(raw.get("evidence_refs", []), episode)
        atomic_payload = raw.get("atomic_evidence_ref")
        if not isinstance(atomic_payload, dict):
            raise ValueError(f"candidate[{index}] atomic_evidence_ref must be an object")
        atomic_refs = self._evidence_refs([atomic_payload], episode)
        if len(atomic_refs) != 1:
            raise ValueError(f"candidate[{index}] atomic_evidence_ref must identify one exact span")
        atomic_evidence_ref = atomic_refs[0]
        if atomic_evidence_ref.span_start is None or atomic_evidence_ref.span_end is None:
            raise ValueError(f"candidate[{index}] atomic_evidence_ref requires an exact span")
        evidence_refs = tuple(dict.fromkeys((*evidence_refs, atomic_evidence_ref)))
        field_evidence_refs = self._field_evidence_refs(
            index,
            raw.get("field_evidence_refs"),
            episode,
            identity_fields=identity_fields,
            value_fields=value_fields,
            evidence_refs=evidence_refs,
        )
        for field_name in (*sorted(f"semantic.{name}" for name in self._SEMANTIC_FIELDS), "transition"):
            if field_evidence_refs.get(field_name) != (atomic_evidence_ref,):
                raise ValueError(f"candidate[{index}] {field_name} must bind only to atomic_evidence_ref")
        raw_scopes = raw.get("suggested_scope_refs", []) or []
        if not isinstance(raw_scopes, list) or any(not isinstance(item, dict) for item in raw_scopes):
            raise ValueError(f"candidate[{index}] suggested_scope_refs must contain only objects")
        scopes = self._canonical_scope_refs(index, raw_scopes, legal_scopes)
        proposal_id = str(
            raw.get("proposal_id") or f"proposal_{stable_hash([episode.episode_id, index, raw], length=32)}"
        )
        if proposal_id in proposal_ids:
            raise ValueError(f"duplicate proposal_id: {proposal_id}")
        related_candidate_refs = self._string_list(
            index,
            "related_candidate_refs",
            raw.get("related_candidate_refs", []),
        )
        if any(reference not in existing_candidates for reference in related_candidate_refs):
            raise ValueError(f"candidate[{index}] related_candidate_refs contains an illegal reference")
        related_items = tuple(existing_candidates[reference] for reference in related_candidate_refs)
        relation = str(semantic.get("relation_to_existing", "unrelated")).strip().casefold()
        if relation in {"corrects", "supersedes", "supplements"}:
            if len(related_items) != 1:
                raise ValueError(f"candidate[{index}] {relation} requires exactly one related active claim")
            related_item = related_items[0]
            if (
                related_item.state.upper() != "ACTIVE"
                or not related_item.slot_id
                or not related_item.claim_id
                or related_item.memory_type != memory_type
            ):
                raise ValueError(f"candidate[{index}] {relation} related target is not one compatible active claim")
        elif relation == "unrelated" and related_items:
            raise ValueError(f"candidate[{index}] unrelated relation cannot declare related targets")
        related_slot_ids = tuple(dict.fromkeys(item.slot_id for item in related_items if item.slot_id))
        related_claim_ids = tuple(dict.fromkeys(item.claim_id for item in related_items if item.claim_id))
        actual_source_role = self._source_role((atomic_evidence_ref,), episode)
        reported_source_role = str(raw.get("source_role") or "").casefold()
        normalized_reported = "assistant" if reported_source_role == "agent" else reported_source_role
        if normalized_reported and normalized_reported != actual_source_role:
            raise ValueError(f"candidate[{index}] source_role does not match referenced evidence")
        speech = str(semantic.get("speech_act", "")).casefold()
        commitment = str(semantic.get("commitment", "")).casefold()
        if (speech in {"confirmation", "correction"} or commitment in {"confirmed", "committed"}) and not evidence_refs:
            raise ValueError(f"candidate[{index}] authoritative semantics require evidence_refs")
        return MemorySemanticProposal(
            proposal_id=proposal_id,
            memory_type=memory_type,
            identity_fields=identity_fields,
            value_fields=value_fields,
            semantic=SemanticAssessment(
                str(semantic.get("speech_act", "observation")),
                str(semantic.get("commitment", "weak")),
                str(semantic.get("temporal_scope", "unspecified")),
                str(semantic.get("relation_to_existing", "unrelated")),
                str(semantic.get("utterance_mode", "unknown")),
                str(semantic.get("attribution", "unknown")),
                str(semantic.get("durability", "unknown")),
                str(semantic.get("modal_force", "unknown")),
                str(semantic.get("atomicity", "unknown")),
            ),
            epistemic_status=EpistemicStatus(str(raw.get("epistemic_status", "INFERRED")).upper()),
            suggested_scope_refs=scopes,
            related_memory_ids=(),
            related_slot_ids=related_slot_ids,
            related_claim_ids=related_claim_ids,
            evidence_refs=evidence_refs,
            field_evidence_refs=field_evidence_refs,
            confidence=confidence,
            extractor_version=self.extractor_version,
            model_id=self.model_id,
            prompt_version=self.prompt_version,
            semantic_contract_version=self.semantic_contract_version,
            atomic_evidence_ref=atomic_evidence_ref,
            metadata={
                "source_role": actual_source_role,
                "source_adapter_id": adapter_id_from_archive(archive),
                "source_session_id": archive.session_id,
                "source_connect": dict(archive.metadata.get("connect", {}) or {}),
                "model_confidence": confidence,
            },
        )

    def _rejection_flags(self, reason: str) -> tuple[str, ...]:
        normalized = reason.casefold()
        flags = []
        forbidden_control = "forbidden control field" in normalized
        if not forbidden_control and (
            "evidence" in normalized or "event_digest" in normalized or "content_hash" in normalized
        ):
            flags.append("evidence_integrity_rejected")
        if "source_role" in normalized:
            flags.append("source_authority_rejected")
        if "scope" in normalized:
            flags.append("scope_authority_rejected")
        if (
            "related_memory" in normalized
            or "related_claim" in normalized
            or "related_slot" in normalized
            or "related_candidate" in normalized
        ):
            flags.append("target_reference_rejected")
        if "target" in normalized or re.search(r"(?:^|[._])[^,\s]*uri(?:s)?(?:$|[,\s])", normalized):
            flags.append("target_reference_rejected")
        if "unknown fields" in normalized or forbidden_control:
            flags.append("forbidden_output_field_rejected")
        if any(marker in normalized for marker in ("action", "delete", "deletion", "operation", "update")):
            flags.append("operation_control_rejected")
        if "revision" in normalized:
            flags.append("revision_control_rejected")
        if "value_fields contains fields outside" in normalized:
            flags.append("candidate_value_schema_rejected")
        if "confidence must be a finite number between 0 and 1" in normalized:
            flags.extend(("candidate_confidence_rejected", "candidate_schema_rejected"))
        return tuple(flags or ("candidate_schema_rejected",))

    def _forbidden_nested_control_paths(self, raw: Mapping[str, Any]) -> tuple[str, ...]:
        """Reject storage-control keys anywhere below the candidate envelope."""

        paths: list[str] = []

        def visit(value: Any, path: str) -> None:
            if isinstance(value, Mapping):
                for key, nested in value.items():
                    key_text = str(key)
                    nested_path = f"{path}.{key_text}" if path else key_text
                    if self._is_forbidden_control_field(key_text):
                        paths.append(nested_path)
                    visit(nested, nested_path)
            elif isinstance(value, list | tuple):
                for item_index, nested in enumerate(value):
                    visit(nested, f"{path}[{item_index}]")

        # Top-level candidate fields keep their existing strict-envelope error.
        # Only their nested contents are recursively inspected here.
        for field_name, value in raw.items():
            visit(value, str(field_name))
        return tuple(dict.fromkeys(sorted(paths)))

    def _is_forbidden_control_field(self, field_name: str) -> bool:
        snake_case = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", field_name.strip())
        normalized = re.sub(r"[^a-z0-9]+", "_", snake_case.casefold()).strip("_")
        tokens = tuple(item for item in normalized.split("_") if item)
        compact = "".join(tokens)
        operation_word = bool(
            "operation" in tokens
            or (
                compact.endswith(("operation", "operations"))
                and compact not in {"cooperation", "cooperations", "interoperation", "teleoperation"}
            )
        )
        action_control = bool(
            any(item in {"action", "actions"} for item in tokens)
            or compact
            in {
                "requestedaction",
                "databaseaction",
                "dbaction",
                "storageaction",
                "memoryaction",
                "contextaction",
                "targetaction",
                "writeaction",
                "deleteaction",
                "updateaction",
            }
        )
        command_suffixes = (
            "action",
            "command",
            "directive",
            "instruction",
            "mutation",
            "operation",
            "payload",
            "request",
            "target",
        )
        delete_control = normalized in {"delete", "deletion"} or (
            compact.startswith(("delete", "deletion")) and compact.endswith(command_suffixes)
        )
        update_control = normalized in {"update", "updates"} or (
            compact.startswith("update") and compact.endswith(command_suffixes)
        )
        return bool(
            normalized in self._FORBIDDEN_CONTROL_FIELDS
            or operation_word
            or action_control
            or delete_control
            or update_control
            or "target" in tokens
            or "revision" in compact
            or compact.endswith(("uri", "uris"))
        )

    def _validated_candidate_confidence(self, index: int, value: Any) -> float:
        if isinstance(value, bool):
            raise ValueError(f"candidate[{index}] confidence must be a finite number between 0 and 1")
        try:
            confidence = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"candidate[{index}] confidence must be a finite number between 0 and 1") from exc
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"candidate[{index}] confidence must be a finite number between 0 and 1")
        return confidence

    def _canonical_scope_refs(
        self,
        index: int,
        payloads: list[dict[str, Any]],
        legal_scopes: Mapping[str, ScopeRef],
    ) -> tuple[ScopeRef, ...]:
        """Validate a model selection, then return only episode-owned scope refs."""

        selected: list[ScopeRef] = []
        for scope_index, payload in enumerate(payloads):
            unknown = set(payload) - self._SCOPE_FIELDS
            if unknown:
                raise ValueError(
                    f"candidate[{index}] suggested_scope_refs[{scope_index}] contains unknown fields: "
                    f"{','.join(sorted(unknown))}"
                )
            parsed = ScopeRef.from_dict(payload)
            canonical = legal_scopes.get(parsed.key)
            if canonical is None:
                raise ValueError(f"candidate[{index}] suggested_scope_refs contains an illegal scope")
            parsed_payload = parsed.to_dict()
            canonical_payload = canonical.to_dict()
            forged = {
                field_name
                for field_name in payload
                if parsed_payload.get(field_name) != canonical_payload.get(field_name)
            }
            if forged:
                raise ValueError(
                    f"candidate[{index}] suggested_scope_refs[{scope_index}] differs from legal scope: "
                    f"{','.join(sorted(forged))}"
                )
            selected.append(canonical)
        return tuple(selected)

    def _validate_semantic_enums(self, index: int, semantic: dict) -> None:
        values = {
            "speech_act": (semantic.get("speech_act", ""), self._SPEECH_VALUES),
            "commitment": (semantic.get("commitment", ""), self._COMMITMENT_VALUES),
            "temporal_scope": (semantic.get("temporal_scope", ""), self._TEMPORAL_VALUES),
            "relation_to_existing": (semantic.get("relation_to_existing", "unrelated"), self._RELATION_VALUES),
            "utterance_mode": (semantic.get("utterance_mode", ""), self._UTTERANCE_VALUES),
            "attribution": (semantic.get("attribution", ""), self._ATTRIBUTION_VALUES),
            "durability": (semantic.get("durability", ""), self._DURABILITY_VALUES),
            "modal_force": (semantic.get("modal_force", ""), self._MODAL_FORCE_VALUES),
            "atomicity": (semantic.get("atomicity", ""), self._ATOMICITY_VALUES),
        }
        for field_name, (raw_value, allowed) in values.items():
            normalized = str(raw_value).strip().casefold()
            if normalized not in allowed:
                raise ValueError(f"candidate[{index}] {field_name} is not allowed: {raw_value}")

    def _string_list(self, index: int, field_name: str, payload: object) -> tuple[str, ...]:
        if payload is None:
            return ()
        if not isinstance(payload, list) or any(not isinstance(item, str) or not item for item in payload):
            raise ValueError(f"candidate[{index}] {field_name} must contain only non-empty strings")
        return tuple(payload)

    def _source_role(self, evidence_refs: tuple[EvidenceRef, ...], episode: EvidenceEpisode) -> str:
        roles = {event.actor.kind for ref in evidence_refs if (event := episode.event(ref.event_id)) is not None}
        if roles == {"user"}:
            return "user"
        if roles == {"system"}:
            return "system"
        if "tool" in roles or "sensor" in roles or "robot" in roles:
            return "tool"
        if "assistant" in roles:
            return "assistant"
        return "unknown"

    def _evidence_refs(self, payload: object, episode: EvidenceEpisode) -> tuple[EvidenceRef, ...]:
        if not isinstance(payload, list):
            raise ValueError("evidence_refs must be a list")
        refs = []
        for item in payload:
            if not isinstance(item, dict) or not item.get("event_id"):
                raise ValueError("each evidence_ref requires event_id")
            unknown = set(item) - {
                "event_id",
                "event_digest",
                "content_hash",
                "content_path",
                "span_start",
                "span_end",
                "quoted_text",
                "quoted_text_hash",
            }
            if unknown:
                raise ValueError(f"evidence_ref contains unknown fields: {','.join(sorted(unknown))}")
            event = episode.event(str(item["event_id"]))
            if event is None:
                raise ValueError(f"evidence_ref event does not exist in episode: {item['event_id']}")
            span_start = int(item["span_start"]) if item.get("span_start") is not None else None
            span_end = int(item["span_end"]) if item.get("span_end") is not None else None
            text = event.text()
            if (span_start is None) != (span_end is None):
                raise ValueError(f"evidence_ref has an incomplete span: {item['event_id']}")
            if (
                span_start is not None
                and span_end is not None
                and (span_start < 0 or span_end <= span_start or span_end > len(text))
            ):
                raise ValueError(f"evidence_ref span is invalid: {item['event_id']}")
            derived = EvidenceRef.from_event(
                event,
                source_uri=episode.source_uris[0] if episode.source_uris else None,
                content_path=str(item.get("content_path") or event.content_path),
                span_start=span_start,
                span_end=span_end,
            )
            if item.get("event_digest") and str(item["event_digest"]) != derived.event_digest:
                raise ValueError(f"evidence_ref event_digest mismatch: {item['event_id']}")
            if item.get("content_hash") and str(item["content_hash"]) != derived.content_hash:
                raise ValueError(f"evidence_ref content_hash mismatch: {item['event_id']}")
            if item.get("quoted_text") is not None and str(item["quoted_text"]) != derived.quoted_text:
                raise ValueError(f"evidence_ref quoted_text mismatch: {item['event_id']}")
            if item.get("quoted_text_hash") and str(item["quoted_text_hash"]) != derived.quoted_text_hash:
                raise ValueError(f"evidence_ref quoted_text_hash mismatch: {item['event_id']}")
            refs.append(derived)
        return tuple(refs)

    def _field_evidence_refs(
        self,
        index: int,
        payload: object,
        episode: EvidenceEpisode,
        *,
        identity_fields: dict,
        value_fields: dict,
        evidence_refs: tuple[EvidenceRef, ...],
    ) -> dict[str, tuple[EvidenceRef, ...]]:
        if not isinstance(payload, dict):
            raise ValueError(f"candidate[{index}] field_evidence_refs must be an object")
        required = {
            *[f"identity.{key}" for key in identity_fields],
            *[f"value.{key}" for key in value_fields],
            *(f"semantic.{field_name}" for field_name in self._SEMANTIC_FIELDS),
            "transition",
        }
        if set(payload) != required:
            missing = required - set(payload)
            unknown = set(payload) - required
            details = [
                *(f"missing:{key}" for key in sorted(missing)),
                *(f"unknown:{key}" for key in sorted(unknown)),
            ]
            raise ValueError(f"candidate[{index}] field_evidence_refs mismatch: {','.join(details)}")
        allowed = set(evidence_refs)
        results = {}
        for field_name in sorted(required):
            refs = self._evidence_refs(payload[field_name], episode)
            if not refs or any(ref not in allowed for ref in refs):
                raise ValueError(f"candidate[{index}] field_evidence_refs invalid for {field_name}")
            results[field_name] = refs
        return results


@dataclass
class FakeMemoryModelProvider:
    """约定 FakeMemoryModelProvider 需要提供的接口。"""

    response: str
    prompts: list[str] | None = None
    is_remote: bool = False

    def complete(self, prompt: str) -> str:
        if self.prompts is not None:
            self.prompts.append(prompt)
        return self.response
