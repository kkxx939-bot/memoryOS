"""Evidence-bounded canonical semantic reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from memoryos.memory.canonical.identity import ResolvedMemoryIdentity
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
        if semantic.speech_act in {SpeechAct.RETRACTION, SpeechAct.REJECTION} and self._authoritative_evidence(
            proposal
        ):
            return SemanticRelation.CORRECTS
        if incoming == current:
            if claim.current.state != "ACTIVE" and active is not None and self._confirmed_current(proposal):
                return SemanticRelation.SUPERSEDES
            return SemanticRelation.DUPLICATE
        if all(key not in current or current[key] == value for key, value in incoming.items()):
            return SemanticRelation.SUPPLEMENTS
        if semantic.speech_act == SpeechAct.CORRECTION and self._authoritative_evidence(proposal):
            return SemanticRelation.CORRECTS
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
        if (
            semantic.speech_act == SpeechAct.CORRECTION
            and active is not None
            and self._authoritative_evidence(proposal)
        ):
            return SemanticRelation.CORRECTS
        if semantic.speech_act in {SpeechAct.PROPOSAL, SpeechAct.EVALUATION_REQUEST} or (
            semantic.commitment in {Commitment.WEAK, Commitment.EXPLORATORY}
            and semantic.temporal_scope == TemporalScope.FUTURE
        ):
            return SemanticRelation.ALTERNATIVE
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
        if self._confirmed_current(proposal) and active is not None:
            return SemanticRelation.SUPERSEDES
        if active is None and not claims:
            return SemanticRelation.UNRELATED
        return SemanticRelation.AMBIGUOUS

    def _supported_suggestion(self, proposal: MemorySemanticProposal) -> bool:
        transition_refs = tuple(proposal.field_evidence_refs.get("transition", ()))
        return bool(
            proposal.metadata.get("semantic_relation_evidence_validated") is True
            and proposal.evidence_refs
            and transition_refs
            and set(transition_refs).issubset(proposal.evidence_refs)
        )

    def _authoritative_evidence(self, proposal: MemorySemanticProposal) -> bool:
        transition_refs = tuple(proposal.field_evidence_refs.get("transition", ()))
        return bool(
            proposal.epistemic_status == EpistemicStatus.EXPLICIT
            and proposal.metadata.get("transition_evidence_validated") is True
            and transition_refs
            and set(transition_refs).issubset(proposal.evidence_refs)
        )

    def _confirmed_current(self, proposal: MemorySemanticProposal) -> bool:
        semantic = proposal.semantic
        assert isinstance(semantic, NormalizedSemanticAssessment)
        return (
            self._authoritative_evidence(proposal)
            and semantic.temporal_scope in {TemporalScope.CURRENT, TemporalScope.UNSPECIFIED}
            and (
                semantic.speech_act in {SpeechAct.CONFIRMATION, SpeechAct.CORRECTION}
                or semantic.commitment == Commitment.CONFIRMED
            )
        )

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
