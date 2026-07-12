"""记忆系统里的准入。"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum

from memoryos.adapters.agent_hooks.sanitizer import ENV_SECRET_RE, INLINE_SECRET_RE, PRIVATE_KEY_RE, SECRET_KEY_RE
from memoryos.memory.canonical.episode import EvidenceEpisode
from memoryos.memory.canonical.evidence import ProposalValidationResult
from memoryos.memory.canonical.proposal import (
    EpistemicStatus,
    MemorySemanticProposal,
    NormalizedSemanticAssessment,
    SemanticRelation,
    TemporalScope,
)
from memoryos.memory.canonical.scope import HIERARCHICAL_SCOPE_KINDS, MemoryScope
from memoryos.memory.canonical.semantic import (
    EligibilityDisposition,
    MemorySemanticNormalizer,
    MemoryTypeEligibilityPolicy,
)
from memoryos.memory.schema import MemoryType, MemoryTypeRegistry

_PRIVATE_PROCESS_RE = re.compile(
    r"(?i)\b(chain of thought|scratchpad|internal reasoning|agent private|内部推理|草稿)\b"
)
_EVIDENCE_INTEGRITY_ERROR_PREFIXES = (
    "unknown_event:",
    "content_path_mismatch:",
    "content_hash_mismatch:",
    "incomplete_span:",
    "invalid_span:",
    "quoted_text_hash_mismatch:",
    "quoted_text_mismatch:",
    "unexpected_quote_without_span:",
    "event_digest_mismatch:",
    "event_schema_mismatch:",
    "tenant_mismatch:",
    "episode_mismatch:",
    "actor_id_mismatch:",
    "actor_kind_mismatch:",
    "actor_role_mismatch:",
    "actor_id_inference_mismatch:",
    "actor_role_inference_mismatch:",
    "occurred_at_mismatch:",
    "ingested_at_mismatch:",
    "sequence_mismatch:",
    "evidence_strength_mismatch:",
    "subject_mismatch:",
    "source_uri_mismatch:",
    "source_role_evidence_mismatch",
    "invalid_field_evidence:",
    "atomic_evidence_ref_not_in_evidence",
    "atomic_evidence_ref_not_validated",
)
_V3_SCORING_BINDINGS = (
    "semantic.utterance_mode",
    "semantic.attribution",
    "semantic.durability",
    "semantic.modal_force",
    "semantic.atomicity",
)


class ProposalAdmissionDecision(str, Enum):
    """保存 ProposalAdmissionDecision 需要的这组数据。"""

    ACCEPT_FOR_RECONCILE = "ACCEPT_FOR_RECONCILE"
    PENDING = "PENDING"
    ARCHIVE_ONLY = "ARCHIVE_ONLY"
    PRIVATE_ONLY = "PRIVATE_ONLY"
    RESTRICTED = "RESTRICTED"
    REJECT = "REJECT"


@dataclass(frozen=True)
class ProposalAdmissionResult:
    """保存 ProposalAdmissionResult 需要的这组数据。"""

    decision: ProposalAdmissionDecision
    reason: str
    admission_score: float = 0.0
    admission_threshold: float = 0.0
    score_components: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AdmissionScoreConfig:
    """Central, testable policy for system-owned admission scoring."""

    weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "source_authority": 0.18,
            "explicitness": 0.15,
            "evidence_coverage": 0.16,
            "field_evidence_completeness": 0.14,
            "identity_confidence": 0.10,
            "temporality_confidence": 0.08,
            "relation_confidence": 0.08,
            "scope_authority": 0.07,
            "model_confidence": 0.04,
        }
    )
    thresholds: Mapping[MemoryType, float] = field(
        default_factory=lambda: {
            MemoryType.PROFILE: 0.78,
            MemoryType.PREFERENCE: 0.76,
            MemoryType.ENTITY: 0.74,
            MemoryType.EVENT: 0.72,
            MemoryType.PROJECT_RULE: 0.82,
            MemoryType.PROJECT_DECISION: 0.82,
            MemoryType.AGENT_EXPERIENCE: 0.76,
        }
    )

    def __post_init__(self) -> None:
        if abs(sum(float(value) for value in self.weights.values()) - 1.0) > 1e-9:
            raise ValueError("admission score weights must sum to 1")


@dataclass(frozen=True)
class AdmissionScore:
    value: float
    threshold: float
    components: Mapping[str, float]

    @property
    def accepted(self) -> bool:
        return self.value >= self.threshold


class SystemAdmissionScorer:
    def __init__(self, config: AdmissionScoreConfig | None = None) -> None:
        self.config = config or AdmissionScoreConfig()

    def score(
        self,
        validation: ProposalValidationResult,
        *,
        memory_type: MemoryType,
        memory_scope: MemoryScope,
        source_role: str,
    ) -> AdmissionScore:
        proposal = validation.proposal
        semantic = proposal.semantic
        assert isinstance(semantic, NormalizedSemanticAssessment)
        required_bindings = {
            *[f"identity.{key}" for key in proposal.identity_fields],
            *[f"value.{key}" for key in proposal.value_fields],
            "semantic.speech_act",
            "semantic.commitment",
            "semantic.temporal_scope",
            "semantic.relation_to_existing",
            "transition",
        }
        if str(getattr(proposal, "semantic_contract_version", "v2")).casefold() == "v3":
            required_bindings.update(_V3_SCORING_BINDINGS)
        bound = sum(bool(proposal.field_evidence_refs.get(name)) for name in required_bindings)
        source = {
            "user": 1.0,
            "system": 1.0,
            "tool": 0.75,
            "assistant": 0.60,
            "agent": 0.60,
        }.get(source_role.casefold(), 0.20)
        explicitness = {
            EpistemicStatus.EXPLICIT: 1.0,
            EpistemicStatus.OBSERVED: 0.82,
            EpistemicStatus.INFERRED: 0.50,
            EpistemicStatus.HYPOTHESIZED: 0.10,
        }[proposal.epistemic_status]
        temporality = 1.0 if semantic.temporal_scope != TemporalScope.UNSPECIFIED else 0.75
        if semantic.relation_to_existing in {SemanticRelation.CORRECTS, SemanticRelation.SUPERSEDES}:
            relation = 1.0 if proposal.metadata.get("semantic_relation_evidence_validated") is True else 0.20
        elif semantic.relation_to_existing in {SemanticRelation.ALTERNATIVE, SemanticRelation.SUPPLEMENTS}:
            relation = 0.90
        else:
            relation = 0.85
        components = {
            "source_authority": source,
            "explicitness": explicitness,
            "evidence_coverage": 1.0 if proposal.evidence_refs and validation.valid else 0.0,
            "field_evidence_completeness": bound / len(required_bindings) if required_bindings else 0.0,
            "identity_confidence": 1.0
            if proposal.identity_fields and not any(name.startswith("identity.") for name in validation.unsupported_fields)
            else 0.0,
            "temporality_confidence": temporality,
            "relation_confidence": relation,
            "scope_authority": 1.0
            if memory_scope.canonical_subject is not None
            and not memory_scope.canonical_subject.inferred
            and not memory_scope.authority.inferred
            else 0.0,
            "model_confidence": proposal.confidence,
        }
        value = sum(self.config.weights[name] * components[name] for name in self.config.weights)
        return AdmissionScore(value, self.config.thresholds[memory_type], components)


class ProposalAdmissionGate:
    """负责 ProposalAdmissionGate 这部分逻辑。"""

    def __init__(
        self,
        registry: MemoryTypeRegistry | None = None,
        scorer: SystemAdmissionScorer | None = None,
        eligibility_policy: MemoryTypeEligibilityPolicy | None = None,
    ) -> None:
        self.registry = registry or MemoryTypeRegistry()
        self.scorer = scorer or SystemAdmissionScorer()
        self.eligibility_policy = eligibility_policy or MemoryTypeEligibilityPolicy()

    def evaluate(
        self,
        validation: ProposalValidationResult,
        *,
        episode: EvidenceEpisode,
        memory_scope: MemoryScope,
        source_role: str,
    ) -> ProposalAdmissionResult:
        """处理 evaluate 这一步。"""

        proposal = validation.proposal
        try:
            memory_type = MemoryType(proposal.memory_type)
            schema = self.registry.get(memory_type)
        except ValueError:
            return ProposalAdmissionResult(ProposalAdmissionDecision.REJECT, "unsupported_memory_schema")
        try:
            memory_scope.validate_tenant(episode.tenant_id)
        except ValueError:
            return ProposalAdmissionResult(ProposalAdmissionDecision.REJECT, "cross_tenant_visibility")
        legal = {scope.key for scope in episode.legal_scope_candidates()}
        suggested = {scope.key for scope in validation.proposal.suggested_scope_refs}
        if not suggested.issubset(legal):
            return ProposalAdmissionResult(ProposalAdmissionDecision.REJECT, "illegal_scope_suggestion")
        if not validation.valid:
            integrity_errors = tuple(
                error
                for error in validation.errors
                if error.startswith(_EVIDENCE_INTEGRITY_ERROR_PREFIXES)
            )
            if integrity_errors:
                return ProposalAdmissionResult(
                    ProposalAdmissionDecision.REJECT,
                    "evidence_integrity_failed:" + ",".join(integrity_errors),
                )
            prefix = (
                "PENDING_MISSING_EVIDENCE"
                if any(
                    error == "missing_evidence"
                    or error.startswith(("missing_field_evidence:", "missing_atomic_evidence_ref"))
                    for error in validation.errors
                )
                else "validation_failed"
            )
            return ProposalAdmissionResult(
                ProposalAdmissionDecision.PENDING,
                f"{prefix}:{','.join(validation.errors)}",
            )
        normalized_semantic = (
            proposal.semantic
            if isinstance(proposal.semantic, NormalizedSemanticAssessment)
            else MemorySemanticNormalizer().normalize(proposal).semantic
        )
        if not isinstance(normalized_semantic, NormalizedSemanticAssessment):
            return ProposalAdmissionResult(ProposalAdmissionDecision.PENDING, "semantic_not_normalized")
        if not self._semantic_schema_safe(proposal, normalized_semantic):
            return ProposalAdmissionResult(
                ProposalAdmissionDecision.PENDING,
                "semantic_schema_pending:" + ",".join(self._semantic_schema_errors(proposal, normalized_semantic)),
            )
        subject = memory_scope.canonical_subject
        if subject is None:
            return ProposalAdmissionResult(ProposalAdmissionDecision.PENDING, "canonical_subject_missing")
        if subject.kind in HIERARCHICAL_SCOPE_KINDS and not subject.parent_path:
            return ProposalAdmissionResult(ProposalAdmissionDecision.PENDING, "scope_hierarchy_missing")
        if subject.inferred or memory_scope.authority.inferred:
            return ProposalAdmissionResult(ProposalAdmissionDecision.PENDING, "scope_authority_inferred")
        text = json.dumps(
            {"identity_fields": dict(proposal.identity_fields), "value_fields": dict(proposal.value_fields)},
            ensure_ascii=False,
            sort_keys=True,
        )
        if self._raw_tool_output(text):
            return ProposalAdmissionResult(ProposalAdmissionDecision.ARCHIVE_ONLY, "raw_tool_output")
        if self._secret_like(text):
            return ProposalAdmissionResult(ProposalAdmissionDecision.RESTRICTED, "secret_or_sensitive_content")
        effective_source_role = self._effective_source_role(proposal, source_role)
        if _PRIVATE_PROCESS_RE.search(text) or effective_source_role in {"agent_private", "internal"}:
            return ProposalAdmissionResult(ProposalAdmissionDecision.PRIVATE_ONLY, "agent_private_process")
        if str(proposal.semantic_contract_version or "v2").casefold() != "v3":
            return ProposalAdmissionResult(
                ProposalAdmissionDecision.PENDING,
                "semantic_contract_v3_required",
            )
        if not schema.claim_identity_keys(dict(proposal.value_fields)):
            return ProposalAdmissionResult(ProposalAdmissionDecision.PENDING, "claim_identity_incomplete")
        normalized_proposal = replace(proposal, semantic=normalized_semantic)
        v3_gate = self._evaluate_v3_semantics(normalized_proposal)
        if v3_gate is not None:
            return v3_gate
        eligibility = self.eligibility_policy.evaluate(
            normalized_proposal,
            memory_type=memory_type,
            schema=schema,
            source_role=effective_source_role,
        )
        if eligibility.disposition != EligibilityDisposition.ELIGIBLE:
            decision = {
                EligibilityDisposition.PENDING: ProposalAdmissionDecision.PENDING,
                EligibilityDisposition.ARCHIVE_ONLY: ProposalAdmissionDecision.ARCHIVE_ONLY,
                EligibilityDisposition.REJECT: ProposalAdmissionDecision.REJECT,
            }[eligibility.disposition]
            return ProposalAdmissionResult(decision, eligibility.reason)
        scoring_validation = replace(
            validation,
            proposal=normalized_proposal,
        )
        score = self.scorer.score(
            scoring_validation,
            memory_type=memory_type,
            memory_scope=memory_scope,
            source_role=effective_source_role,
        )
        if not score.accepted:
            return ProposalAdmissionResult(
                ProposalAdmissionDecision.PENDING,
                "system_admission_score_below_threshold",
                score.value,
                score.threshold,
                score.components,
            )
        return ProposalAdmissionResult(
            ProposalAdmissionDecision.ACCEPT_FOR_RECONCILE,
            "validated",
            score.value,
            score.threshold,
            score.components,
        )

    def _semantic_schema_safe(
        self,
        proposal: MemorySemanticProposal,
        semantic: NormalizedSemanticAssessment,
    ) -> bool:
        if str(getattr(proposal, "semantic_contract_version", "v2") or "v2").casefold() == "v3":
            return semantic.schema_safe
        return not self._semantic_schema_errors(proposal, semantic)

    def _semantic_schema_errors(
        self,
        proposal: MemorySemanticProposal,
        semantic: NormalizedSemanticAssessment,
    ) -> tuple[str, ...]:
        if str(getattr(proposal, "semantic_contract_version", "v2") or "v2").casefold() == "v3":
            return semantic.schema_errors
        errors: list[str] = []
        for field_name in ("speech_act", "commitment", "temporal_scope", "relation_to_existing"):
            value = str(getattr(getattr(semantic, field_name), "value", getattr(semantic, field_name))).upper()
            if value in {"UNKNOWN", "AMBIGUOUS", "SCHEMA_MISMATCH"}:
                errors.append(f"semantic_{field_name}_{value.lower()}")
        return tuple(errors)

    def _effective_source_role(self, proposal: MemorySemanticProposal, fallback: str) -> str:
        if str(getattr(proposal, "semantic_contract_version", "v2") or "v2").casefold() != "v3":
            return str(fallback).strip().casefold()
        atomic = getattr(proposal, "atomic_evidence_ref", None)
        actor_kind = str(getattr(atomic, "actor_kind", "") or "").strip().casefold()
        return actor_kind or "unknown"

    def _evaluate_v3_semantics(
        self,
        proposal: MemorySemanticProposal,
    ) -> ProposalAdmissionResult | None:
        if str(getattr(proposal, "semantic_contract_version", "v2") or "v2").casefold() != "v3":
            return None
        metadata = dict(getattr(proposal, "metadata", {}) or {})
        if metadata.get("atomic_evidence_validated") is not True or metadata.get(
            "semantic_contract_validated"
        ) is not True:
            return ProposalAdmissionResult(
                ProposalAdmissionDecision.PENDING,
                "semantic_v3_contract_not_validated",
            )
        semantic = proposal.semantic

        def value(field_name: str) -> str:
            raw = getattr(semantic, field_name, "UNKNOWN")
            return str(getattr(raw, "value", raw) or "").strip().upper()

        utterance = value("utterance_mode")
        attribution = value("attribution")
        durability = value("durability")
        atomicity = value("atomicity")
        speech_act = value("speech_act")
        relation = value("relation_to_existing")

        if (
            utterance in {"QUESTION", "HYPOTHETICAL"}
            or attribution == "QUOTED"
            or durability == "TRANSIENT"
        ):
            return ProposalAdmissionResult(
                ProposalAdmissionDecision.ARCHIVE_ONLY,
                "semantic_v3_non_durable_utterance",
            )
        if (
            utterance in {"MIXED", "UNKNOWN", "SCHEMA_MISMATCH"}
            or attribution in {"THIRD_PARTY", "MIXED", "UNKNOWN", "SCHEMA_MISMATCH"}
            or durability in {"UNKNOWN", "SCHEMA_MISMATCH"}
            or atomicity in {"COMPOUND", "UNKNOWN", "SCHEMA_MISMATCH"}
        ):
            return ProposalAdmissionResult(
                ProposalAdmissionDecision.PENDING,
                "semantic_v3_ambiguous_or_compound",
            )
        if (
            utterance not in {"ASSERTION", "DIRECTIVE"}
            or attribution != "SOURCE_ACTOR"
            or durability != "DURABLE"
            or atomicity != "ATOMIC"
        ):
            return ProposalAdmissionResult(
                ProposalAdmissionDecision.PENDING,
                "semantic_v3_not_active_eligible",
            )
        if speech_act in {"PROPOSAL", "EVALUATION_REQUEST"} or relation in {
            "ALTERNATIVE",
            "CONTRADICTS",
        }:
            return ProposalAdmissionResult(
                ProposalAdmissionDecision.PENDING,
                "semantic_v3_nonfinal_relation_requires_review",
            )
        return None

    def _secret_like(self, text: str) -> bool:
        return bool(
            PRIVATE_KEY_RE.search(text)
            or ENV_SECRET_RE.search(text)
            or INLINE_SECRET_RE.search(text)
            or ("<redacted" in text.casefold() and SECRET_KEY_RE.search(text))
            or re.search(r"(?i)\b(authorization\s*:|cookie\s*:)", text)
        )

    def _raw_tool_output(self, text: str) -> bool:
        normalized = text.casefold()
        return any(
            marker in normalized
            for marker in (
                "traceback (most recent call last)",
                "assertionerror",
                "exit code:",
                "stdout:",
                "stderr:",
            )
        )
