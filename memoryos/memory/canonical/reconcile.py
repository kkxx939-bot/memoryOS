"""Evidence-bounded canonical semantic reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from memoryos.memory.canonical.identity import ResolvedMemoryIdentity, canonical_identity_value
from memoryos.memory.canonical.proposal import (
    Commitment,
    EpistemicStatus,
    MemorySemanticProposal,
    NormalizedSemanticAssessment,
    SemanticRelation,
    SpeechAct,
    TemporalScope,
)
from memoryos.memory.canonical.state import MemoryClaim, MemorySlot

_REPLACEMENT_RELATIONS = {SemanticRelation.CORRECTS, SemanticRelation.SUPERSEDES}
_NON_CORE_VALUE_FIELDS = frozenset(
    {
        "title",
        "display_name",
        "summary",
        "details",
        "rationale",
        "reason",
        "decision",
        "rule",
        "source_text",
        "display_text",
        "source_wording",
        "evidence",
    }
)
_APPLICABILITY_FIELDS = (
    "environment",
    "device",
    "activity",
    "valid_time",
    "condition",
    "conditions",
    "exception",
    "exceptions",
    "applicability_qualifier",
)
_AUTHORITATIVE_SOURCE_ROLES = frozenset({"user", "system"})


@dataclass(frozen=True)
class ReconciliationResult:
    relation: SemanticRelation
    slot: MemorySlot | None
    claim: MemoryClaim | None
    active_claim: MemoryClaim | None
    claims: tuple[MemoryClaim, ...]
    deterministic: bool = True
    pending_reason: str = ""
    historical_only: bool = False

    @property
    def transition_allowed(self) -> bool:
        return self.relation not in {
            SemanticRelation.UNKNOWN,
            SemanticRelation.AMBIGUOUS,
            SemanticRelation.SCHEMA_MISMATCH,
        }


class AmbiguousSemanticReconciler(Protocol):
    def reconcile_relation(
        self,
        proposal: MemorySemanticProposal,
        active_claim: MemoryClaim | None,
        claims: tuple[MemoryClaim, ...],
    ) -> SemanticRelation: ...


class MemorySemanticReconciler:
    """Derive relations from state and validated semantic signals."""

    def __init__(self, ambiguous_reconciler: AmbiguousSemanticReconciler | None = None) -> None:
        self.ambiguous_reconciler = ambiguous_reconciler

    def reconcile(
        self,
        proposal: MemorySemanticProposal,
        identity: ResolvedMemoryIdentity,
        *,
        slot: MemorySlot | None,
        claims: tuple[MemoryClaim, ...],
    ) -> ReconciliationResult:
        if slot is not None:
            slot.validate_claims(claims)
        by_id = {item.claim_id: item for item in claims}
        claim = by_id.get(identity.claim_id)
        active = by_id.get(slot.active_claim_id) if slot is not None and slot.active_claim_id else None
        semantic = proposal.semantic
        if not isinstance(semantic, NormalizedSemanticAssessment):
            return ReconciliationResult(
                SemanticRelation.SCHEMA_MISMATCH,
                slot,
                claim,
                active,
                claims,
                pending_reason="semantic_not_normalized",
            )
        if (
            semantic.relation_to_existing
            in {
                SemanticRelation.UNKNOWN,
                SemanticRelation.AMBIGUOUS,
                SemanticRelation.SCHEMA_MISMATCH,
            }
            or not semantic.schema_safe
        ):
            return ReconciliationResult(
                semantic.relation_to_existing
                if semantic.relation_to_existing
                in {SemanticRelation.UNKNOWN, SemanticRelation.AMBIGUOUS, SemanticRelation.SCHEMA_MISMATCH}
                else SemanticRelation.SCHEMA_MISMATCH,
                slot,
                claim,
                active,
                claims,
                pending_reason="semantic_schema_pending",
            )

        historical_only = active is not None and self._effective_time(proposal) < self._effective_time(active)
        if claim is not None:
            relation = self._same_claim_relation(proposal, claim, active)
        elif slot is None or not claims:
            relation = SemanticRelation.UNRELATED
        else:
            relation = self._different_claim_relation(proposal, active, claims)

        deterministic = True
        pending_detail = ""
        if relation == SemanticRelation.AMBIGUOUS and self.ambiguous_reconciler is not None:
            suggested = self.ambiguous_reconciler.reconcile_relation(proposal, active, claims)
            deterministic = False
            pending_detail = f":suggested_{suggested.value.lower()}"
        pending = (
            "relation_requires_confirmation" + pending_detail
            if relation
            in {
                SemanticRelation.UNKNOWN,
                SemanticRelation.AMBIGUOUS,
                SemanticRelation.SCHEMA_MISMATCH,
            }
            else ""
        )
        return ReconciliationResult(relation, slot, claim, active, claims, deterministic, pending, historical_only)

    def _same_claim_relation(
        self,
        proposal: MemorySemanticProposal,
        claim: MemoryClaim,
        active: MemoryClaim | None,
    ) -> SemanticRelation:
        incoming = dict(proposal.value_fields)
        current = dict(claim.current.value_fields)
        semantic = proposal.semantic
        assert isinstance(semantic, NormalizedSemanticAssessment)
        replacement = self._validated_replacement_relation(proposal, active, incoming_claim=claim)
        if replacement is not None:
            return replacement
        if semantic.speech_act in {SpeechAct.RETRACTION, SpeechAct.REJECTION} and self._authoritative_evidence(
            proposal
        ):
            return SemanticRelation.CORRECTS
        if incoming == current:
            if semantic.relation_to_existing == SemanticRelation.SUPPLEMENTS and self._supported_suggestion(proposal):
                return SemanticRelation.SUPPLEMENTS
            return SemanticRelation.DUPLICATE
        incoming_core = self._core_value_fields(incoming)
        current_core = self._core_value_fields(current)
        if incoming_core and incoming_core == current_core:
            if semantic.relation_to_existing == SemanticRelation.SUPPLEMENTS and self._supported_suggestion(proposal):
                return SemanticRelation.SUPPLEMENTS
            return SemanticRelation.DUPLICATE
        if incoming_core and all(
            key not in current_core or current_core[key] == value for key, value in incoming_core.items()
        ):
            return SemanticRelation.SUPPLEMENTS
        return SemanticRelation.AMBIGUOUS

    def _different_claim_relation(
        self,
        proposal: MemorySemanticProposal,
        active: MemoryClaim | None,
        claims: tuple[MemoryClaim, ...],
    ) -> SemanticRelation:
        semantic = proposal.semantic
        assert isinstance(semantic, NormalizedSemanticAssessment)
        related = set(proposal.all_related_memory_ids)
        related_claims = tuple(
            claim for claim in claims if claim.claim_id in related or claim.uri in related or claim.slot_id in related
        )
        suggestion = semantic.relation_to_existing
        if semantic.speech_act in {SpeechAct.PROPOSAL, SpeechAct.EVALUATION_REQUEST} or (
            semantic.commitment in {Commitment.WEAK, Commitment.EXPLORATORY}
            and semantic.temporal_scope == TemporalScope.FUTURE
        ):
            return SemanticRelation.ALTERNATIVE
        if suggestion == SemanticRelation.ALTERNATIVE and self._supported_suggestion(proposal):
            return SemanticRelation.ALTERNATIVE
        replacement = self._validated_replacement_relation(proposal, active)
        if replacement is not None:
            return replacement
        if (
            suggestion == SemanticRelation.CONTRADICTS
            and active is not None
            and related_claims
            and self._supported_suggestion(proposal)
        ):
            return SemanticRelation.CONTRADICTS
        if (
            suggestion == SemanticRelation.SUPPLEMENTS
            and (related_claims or proposal.memory_type == "agent_experience")
            and self._supported_suggestion(proposal)
        ):
            return SemanticRelation.SUPPLEMENTS
        if active is None and not claims:
            return SemanticRelation.UNRELATED
        return SemanticRelation.AMBIGUOUS

    def _supported_suggestion(self, proposal: MemorySemanticProposal) -> bool:
        transition_refs = tuple(proposal.field_evidence_refs.get("transition", ()))
        relation_refs = tuple(proposal.field_evidence_refs.get("semantic.relation_to_existing", ()))
        return bool(
            proposal.metadata.get("semantic_relation_evidence_validated") is True
            and proposal.evidence_refs
            and transition_refs
            and relation_refs
            and set(transition_refs).issubset(proposal.evidence_refs)
            and set(relation_refs).issubset(proposal.evidence_refs)
        )

    def _authoritative_evidence(self, proposal: MemorySemanticProposal) -> bool:
        transition_refs = tuple(proposal.field_evidence_refs.get("transition", ()))
        return bool(
            proposal.epistemic_status == EpistemicStatus.EXPLICIT
            and proposal.metadata.get("transition_evidence_validated") is True
            and transition_refs
            and set(transition_refs).issubset(proposal.evidence_refs)
            and self._source_authoritative(proposal, transition_refs)
        )

    def _validated_replacement_relation(
        self,
        proposal: MemorySemanticProposal,
        active: MemoryClaim | None,
        *,
        incoming_claim: MemoryClaim | None = None,
    ) -> SemanticRelation | None:
        semantic = proposal.semantic
        assert isinstance(semantic, NormalizedSemanticAssessment)
        relation = semantic.relation_to_existing
        if relation not in _REPLACEMENT_RELATIONS or active is None:
            return None
        if incoming_claim is not None and incoming_claim.claim_id == active.claim_id:
            return None
        if semantic.temporal_scope != TemporalScope.CURRENT:
            return None
        if not self._explicitly_targets_active(proposal, active):
            return None
        if self._applicability_conflicts(proposal, active):
            return None
        relation_refs = tuple(proposal.field_evidence_refs.get("semantic.relation_to_existing", ()))
        if not (
            self._authoritative_evidence(proposal)
            and proposal.metadata.get("semantic_relation_evidence_validated") is True
            and proposal.metadata.get("replacement_evidence_validated") is True
            and relation_refs
            and set(relation_refs).issubset(proposal.evidence_refs)
        ):
            return None
        return relation

    def _explicitly_targets_active(self, proposal: MemorySemanticProposal, active: MemoryClaim) -> bool:
        explicit_claim_targets = {
            *proposal.related_claim_ids,
            *proposal.related_memory_ids,
        }
        return active.claim_id in explicit_claim_targets or active.uri in explicit_claim_targets

    def _source_authoritative(
        self,
        proposal: MemorySemanticProposal,
        transition_refs: tuple[object, ...],
    ) -> bool:
        declared = str(proposal.metadata.get("source_role") or "").strip().casefold()
        if declared and declared not in _AUTHORITATIVE_SOURCE_ROLES:
            return False
        actor_kinds = {
            str(actor_kind).strip().casefold()
            for ref in transition_refs
            if (actor_kind := getattr(ref, "actor_kind", None))
        }
        if actor_kinds and not actor_kinds.issubset(_AUTHORITATIVE_SOURCE_ROLES):
            return False
        return bool(declared in _AUTHORITATIVE_SOURCE_ROLES or actor_kinds)

    def _applicability_conflicts(self, proposal: MemorySemanticProposal, active: MemoryClaim) -> bool:
        incoming = proposal.value_fields
        current = active.current.value_fields
        for field_name in _APPLICABILITY_FIELDS:
            incoming_value = incoming.get(field_name)
            current_value = current.get(field_name)
            incoming_present = self._present(incoming_value)
            current_present = self._present(current_value)
            if incoming_present != current_present:
                return True
            if not incoming_present:
                continue
            if canonical_identity_value(incoming_value) != canonical_identity_value(current_value):
                return True
        return False

    def _core_value_fields(self, values: dict) -> dict:
        return {
            key: canonical_identity_value(value)
            for key, value in values.items()
            if key not in _NON_CORE_VALUE_FIELDS
        }

    def _present(self, value: object) -> bool:
        return value is not None and value != "" and value != () and value != [] and value != {}

    def _effective_time(self, value: MemorySemanticProposal | MemoryClaim) -> datetime:
        if isinstance(value, MemoryClaim):
            raw = value.current.valid_from
        else:
            raw = str(
                value.metadata.get("effective_at")
                or value.metadata.get("valid_from")
                or value.metadata.get("occurred_at")
                or ""
            )
        if not raw:
            return datetime.max.replace(tzinfo=timezone.utc)
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return datetime.max.replace(tzinfo=timezone.utc)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
