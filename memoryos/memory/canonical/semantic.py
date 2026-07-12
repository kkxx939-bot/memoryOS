"""Fail-closed semantic normalization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import TypeVar

from memoryos.memory.canonical.proposal import (
    Atomicity,
    Attribution,
    Commitment,
    Durability,
    EpistemicStatus,
    MemorySemanticProposal,
    ModalForce,
    NormalizedSemanticAssessment,
    SemanticRelation,
    SpeechAct,
    TemporalScope,
    UtteranceMode,
)
from memoryos.memory.schema import MemoryType, MemoryTypeSchema

SemanticEnum = TypeVar("SemanticEnum", bound=Enum)


class EligibilityDisposition(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    PENDING = "PENDING"
    ARCHIVE_ONLY = "ARCHIVE_ONLY"
    REJECT = "REJECT"


@dataclass(frozen=True)
class EligibilityResult:
    disposition: EligibilityDisposition
    reason: str = ""


_AUTHORITATIVE_MEMORY_TYPES = frozenset(
    {
        MemoryType.PROFILE,
        MemoryType.PREFERENCE,
        MemoryType.PROJECT_RULE,
        MemoryType.PROJECT_DECISION,
    }
)


def project_rule_semantics_consistent(proposal: MemorySemanticProposal) -> bool:
    semantic = proposal.semantic
    raw_force = getattr(semantic, "modal_force", "UNKNOWN")
    force = str(getattr(raw_force, "value", raw_force) or "").strip().upper()
    allowed_values = {
        "REQUIRE": {"REQUIRE", "REQUIRED"},
        "FORBID": {"FORBID", "FORBIDDEN"},
        "ALLOW": {"ALLOW", "ALLOWED"},
        "PREFER": {"PREFER", "PREFERRED"},
        "DISCOURAGE": {"DISCOURAGE", "DISCOURAGED"},
        "CONDITIONAL_REQUIRE": {"CONDITIONAL_REQUIRE", "REQUIRE", "REQUIRED"},
        "CONDITIONAL_FORBID": {"CONDITIONAL_FORBID", "FORBID", "FORBIDDEN"},
    }
    expected = allowed_values.get(force)
    if expected is None:
        return False
    fields = dict(proposal.value_fields)

    def present(value: object) -> bool:
        return value is not None and value != "" and value != () and value != [] and value != {}

    declared = [
        str(fields[field_name]).strip().upper()
        for field_name in ("constraint_polarity", "polarity")
        if present(fields.get(field_name))
    ]
    canonical = str(fields.get("canonical_value") or "").strip().upper()
    known_values = {item for values in allowed_values.values() for item in values}
    if canonical in known_values:
        declared.append(canonical)
    if not declared or any(item not in expected for item in declared):
        return False
    has_condition = any(
        present(fields.get(field_name))
        for field_name in (
            "condition",
            "conditions",
            "exception",
            "exceptions",
            "applicability_qualifier",
        )
    )
    conditional = force in {"CONDITIONAL_REQUIRE", "CONDITIONAL_FORBID"}
    return has_condition if conditional else not has_condition


class MemoryTypeEligibilityPolicy:
    """One structural eligibility matrix shared by admission and transition."""

    def evaluate(
        self,
        proposal: MemorySemanticProposal,
        *,
        memory_type: MemoryType,
        schema: MemoryTypeSchema,
        source_role: str,
    ) -> EligibilityResult:
        semantic = proposal.semantic
        if not isinstance(semantic, NormalizedSemanticAssessment):
            return EligibilityResult(EligibilityDisposition.PENDING, "semantic_not_normalized")
        if proposal.epistemic_status == EpistemicStatus.HYPOTHESIZED:
            return EligibilityResult(EligibilityDisposition.PENDING, "hypothesis_requires_confirmation")

        role = str(source_role or "").strip().casefold()
        if memory_type in _AUTHORITATIVE_MEMORY_TYPES and role not in {"user", "system"}:
            return EligibilityResult(
                EligibilityDisposition.PENDING,
                "semantic_v3_source_not_authoritative",
            )
        if role == "user" and not schema.allow_user_source:
            return EligibilityResult(EligibilityDisposition.REJECT, "user_source_not_allowed")
        if role in {"assistant", "agent"} and not schema.allow_assistant_source:
            return EligibilityResult(
                EligibilityDisposition.PENDING,
                "assistant_source_not_authoritative",
            )
        if role == "tool":
            if not schema.allow_tool_source:
                return EligibilityResult(EligibilityDisposition.ARCHIVE_ONLY, "tool_source_not_allowed")
            if proposal.epistemic_status != EpistemicStatus.OBSERVED:
                return EligibilityResult(EligibilityDisposition.ARCHIVE_ONLY, "tool_claim_not_observed")
        elif role not in {"user", "system", "assistant", "agent"}:
            return EligibilityResult(
                EligibilityDisposition.PENDING,
                "semantic_v3_source_not_authoritative",
            )

        if memory_type in _AUTHORITATIVE_MEMORY_TYPES:
            if (
                proposal.epistemic_status != EpistemicStatus.EXPLICIT
                or semantic.commitment != Commitment.CONFIRMED
            ):
                return EligibilityResult(
                    EligibilityDisposition.PENDING,
                    "semantic_v3_authoritative_commitment_pending",
                )
            if semantic.temporal_scope != TemporalScope.CURRENT:
                return EligibilityResult(
                    EligibilityDisposition.PENDING,
                    "semantic_v3_authoritative_temporality_pending",
                )
        else:
            allowed_temporal = (
                {TemporalScope.PAST, TemporalScope.CURRENT}
                if memory_type in {MemoryType.EVENT, MemoryType.AGENT_EXPERIENCE}
                else {TemporalScope.CURRENT}
            )
            if semantic.temporal_scope not in allowed_temporal:
                return EligibilityResult(
                    EligibilityDisposition.PENDING,
                    "semantic_v3_non_authoritative_temporality_pending",
                )

        is_retraction = semantic.speech_act in {SpeechAct.RETRACTION, SpeechAct.REJECTION}
        if (
            memory_type == MemoryType.PREFERENCE
            and not is_retraction
            and semantic.modal_force not in {ModalForce.PREFER, ModalForce.DISCOURAGE}
        ):
            return EligibilityResult(
                EligibilityDisposition.PENDING,
                "preference_modal_force_inconsistent",
            )
        if (
            memory_type == MemoryType.PROJECT_RULE
            and not is_retraction
            and not project_rule_semantics_consistent(proposal)
        ):
            return EligibilityResult(
                EligibilityDisposition.PENDING,
                "project_rule_semantic_inconsistent",
            )
        return EligibilityResult(EligibilityDisposition.ELIGIBLE)


class MemorySemanticNormalizer:
    """Normalize only explicit, versioned aliases; preserve all uncertainty."""

    VERSION = "memory_semantic_alias_v3"

    _SPEECH = {
        "observation": SpeechAct.OBSERVATION,
        "proposal": SpeechAct.PROPOSAL,
        "recommendation": SpeechAct.PROPOSAL,
        "future_option": SpeechAct.PROPOSAL,
        "possible_alternative": SpeechAct.PROPOSAL,
        "exploratory_alternative": SpeechAct.PROPOSAL,
        "under_consideration": SpeechAct.EVALUATION_REQUEST,
        "evaluation_request": SpeechAct.EVALUATION_REQUEST,
        "confirmation": SpeechAct.CONFIRMATION,
        "correction": SpeechAct.CORRECTION,
        "retraction": SpeechAct.RETRACTION,
        "rejection": SpeechAct.REJECTION,
        "unknown": SpeechAct.UNKNOWN,
        "schema_mismatch": SpeechAct.SCHEMA_MISMATCH,
    }
    _COMMITMENT = {
        "weak": Commitment.WEAK,
        "possible": Commitment.WEAK,
        "exploratory": Commitment.EXPLORATORY,
        "exploratory_alternative": Commitment.EXPLORATORY,
        "future_option": Commitment.EXPLORATORY,
        "recommendation": Commitment.EXPLORATORY,
        "intended": Commitment.INTENDED,
        "plan": Commitment.INTENDED,
        "confirmed": Commitment.CONFIRMED,
        "committed": Commitment.CONFIRMED,
        "unknown": Commitment.UNKNOWN,
        "schema_mismatch": Commitment.SCHEMA_MISMATCH,
    }
    _TEMPORAL = {item.value.lower(): item for item in TemporalScope}
    _RELATION = {item.value.lower(): item for item in SemanticRelation}
    _RELATION.update(
        {"possible_alternative": SemanticRelation.ALTERNATIVE, "exploratory_alternative": SemanticRelation.ALTERNATIVE}
    )
    _UTTERANCE = {item.value.lower(): item for item in UtteranceMode}
    _ATTRIBUTION = {item.value.lower(): item for item in Attribution}
    _DURABILITY = {item.value.lower(): item for item in Durability}
    _MODAL_FORCE = {item.value.lower(): item for item in ModalForce}
    _ATOMICITY = {item.value.lower(): item for item in Atomicity}

    def normalize(self, proposal: MemorySemanticProposal) -> MemorySemanticProposal:
        semantic = proposal.semantic
        if isinstance(semantic, NormalizedSemanticAssessment):
            metadata = {
                **dict(proposal.metadata),
                "semantic_normalization_version": self.VERSION,
                "semantic_normalization_errors": list(semantic.schema_errors),
            }
            return replace(proposal, metadata=metadata)
        normalized = NormalizedSemanticAssessment(
            speech_act=self._map(self._SPEECH, semantic.speech_act, SpeechAct.UNKNOWN, SpeechAct.SCHEMA_MISMATCH),
            commitment=self._map(
                self._COMMITMENT,
                semantic.commitment,
                Commitment.UNKNOWN,
                Commitment.SCHEMA_MISMATCH,
            ),
            temporal_scope=self._map(
                self._TEMPORAL,
                semantic.temporal_scope,
                TemporalScope.UNKNOWN,
                TemporalScope.SCHEMA_MISMATCH,
            ),
            relation_to_existing=self._map(
                self._RELATION,
                semantic.relation_to_existing,
                SemanticRelation.UNKNOWN,
                SemanticRelation.SCHEMA_MISMATCH,
            ),
            utterance_mode=self._map(
                self._UTTERANCE,
                semantic.utterance_mode,
                UtteranceMode.UNKNOWN,
                UtteranceMode.SCHEMA_MISMATCH,
            ),
            attribution=self._map(
                self._ATTRIBUTION,
                semantic.attribution,
                Attribution.UNKNOWN,
                Attribution.SCHEMA_MISMATCH,
            ),
            durability=self._map(
                self._DURABILITY,
                semantic.durability,
                Durability.UNKNOWN,
                Durability.SCHEMA_MISMATCH,
            ),
            modal_force=self._map(
                self._MODAL_FORCE,
                semantic.modal_force,
                ModalForce.UNKNOWN,
                ModalForce.SCHEMA_MISMATCH,
            ),
            atomicity=self._map(
                self._ATOMICITY,
                semantic.atomicity,
                Atomicity.UNKNOWN,
                Atomicity.SCHEMA_MISMATCH,
            ),
        )
        metadata = {
            **dict(proposal.metadata),
            "semantic_normalization_version": self.VERSION,
            "semantic_normalization_errors": list(normalized.schema_errors),
        }
        return replace(proposal, semantic=normalized, metadata=metadata)

    def _map(
        self,
        mapping: Mapping[str, SemanticEnum],
        value: str,
        unknown: SemanticEnum,
        mismatch: SemanticEnum,
    ) -> SemanticEnum:
        raw = value.value if isinstance(value, Enum) else value
        normalized = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
        if not normalized:
            return unknown
        return mapping.get(normalized, mismatch)
