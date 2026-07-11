"""The sole canonical Claim state transition policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

from memoryos.core.time import utc_now
from memoryos.memory.canonical.identity import ResolvedMemoryIdentity
from memoryos.memory.canonical.proposal import (
    Commitment,
    EpistemicStatus,
    MemorySemanticProposal,
    NormalizedSemanticAssessment,
    SemanticRelation,
    SpeechAct,
)
from memoryos.memory.canonical.reconcile import ReconciliationResult
from memoryos.memory.canonical.state import MemoryClaim, MemoryRevision, MemorySlot, TransitionProfile, profile_for


class PendingSemanticReconciliation(ValueError):
    """Fail closed without producing a canonical state change."""

    def __init__(self, relation: SemanticRelation, reason: str) -> None:
        self.relation = relation
        self.reason = reason
        super().__init__(f"semantic reconciliation pending: {relation.value}:{reason}")


@dataclass(frozen=True)
class MemoryStateTransition:
    slot: MemorySlot
    claims: tuple[MemoryClaim, ...]
    changed_claim_ids: tuple[str, ...]
    expected_revisions: Mapping[str, int]
    relation: SemanticRelation

    def __post_init__(self) -> None:
        object.__setattr__(self, "claims", tuple(self.claims))
        object.__setattr__(self, "changed_claim_ids", tuple(self.changed_claim_ids))
        object.__setattr__(
            self,
            "expected_revisions",
            MappingProxyType({str(uri): int(revision) for uri, revision in sorted(self.expected_revisions.items())}),
        )


class MemoryTransitionPolicy:
    """Generate the unique legal state result for one Slot."""

    VERSION = "memory_transition_v2"

    def apply(
        self,
        proposal: MemorySemanticProposal,
        identity: ResolvedMemoryIdentity,
        reconciliation: ReconciliationResult,
    ) -> MemoryStateTransition:
        semantic = proposal.semantic
        if not isinstance(semantic, NormalizedSemanticAssessment):
            raise PendingSemanticReconciliation(SemanticRelation.SCHEMA_MISMATCH, "semantic_not_normalized")
        if not reconciliation.transition_allowed:
            raise PendingSemanticReconciliation(reconciliation.relation, reconciliation.pending_reason)
        if reconciliation.slot is not None:
            reconciliation.slot.validate_claims(reconciliation.claims)

        profile = profile_for(proposal.memory_type)
        claims = list(reconciliation.claims)
        expected = {claim.uri: claim.latest_revision.revision for claim in claims}
        slot = reconciliation.slot or MemorySlot(
            slot_id=identity.slot_id,
            uri=identity.slot_uri,
            memory_type=proposal.memory_type,
            identity_fields=identity.slot_identity,
            scope_keys=identity.scope_keys,
            identity_algorithm_version=identity.identity_algorithm_version,
            canonical_subject_key=identity.canonical_subject_key,
            canonical_subject=identity.canonical_subject,
        )
        expected[slot.uri] = slot.revision
        target = reconciliation.claim
        target_state = self._target_state(profile, proposal, reconciliation)
        changed: list[str] = []

        if target is None:
            revision = self._revision(
                1,
                target_state,
                proposal,
                reconciliation.relation,
                self._proposal_qualifiers(semantic, target_state, reconciliation.historical_only),
            )
            target = MemoryClaim(
                identity.claim_id,
                identity.claim_uri,
                identity.slot_id,
                identity.canonical_value,
                profile,
                (revision,),
                identity.identity_algorithm_version,
                identity.canonical_subject_key,
            )
            claims.append(target)
            expected[target.uri] = 0
            changed.append(target.claim_id)
        else:
            next_state, qualifiers = self._next_existing_state(
                target,
                target_state,
                semantic,
                reconciliation.relation,
                reconciliation.historical_only,
            )
            if (
                reconciliation.relation != SemanticRelation.DUPLICATE
                or next_state != target.current.state
                or qualifiers != dict(target.current.qualifiers)
                or dict(target.current.value_fields) != dict(proposal.value_fields)
            ):
                target = target.with_revision(
                    self._revision(
                        target.latest_revision.revision + 1,
                        next_state,
                        proposal,
                        reconciliation.relation,
                        qualifiers,
                    )
                )
                claims = [target if item.claim_id == target.claim_id else item for item in claims]
                changed.append(target.claim_id)

        active_claim_id = slot.active_claim_id
        if target.current.state == "ACTIVE":
            current_active = next((claim for claim in claims if claim.claim_id == active_claim_id), None)
            if current_active is not None and current_active.claim_id != target.claim_id:
                allowed_replacements = {SemanticRelation.CORRECTS, SemanticRelation.SUPERSEDES}
                if profile in {TransitionProfile.OBSERVATIONAL, TransitionProfile.EXPERIENCE}:
                    allowed_replacements.add(SemanticRelation.SUPPLEMENTS)
                if reconciliation.relation not in allowed_replacements:
                    raise PendingSemanticReconciliation(
                        SemanticRelation.AMBIGUOUS,
                        "active_replacement_requires_corrects_or_supersedes",
                    )
                claims = [
                    self._supersede(claim, target, proposal) if claim.claim_id == current_active.claim_id else claim
                    for claim in claims
                ]
                changed.append(current_active.claim_id)
            active_claim_id = target.claim_id
        elif slot.active_claim_id == target.claim_id:
            active_claim_id = None

        claim_ids = tuple(dict.fromkeys((*slot.claim_ids, *(claim.claim_id for claim in claims))))
        slot_changed = claim_ids != slot.claim_ids or active_claim_id != slot.active_claim_id or bool(changed)
        slot = replace(
            slot,
            claim_ids=claim_ids,
            active_claim_id=active_claim_id,
            revision=slot.revision + (1 if slot_changed else 0),
        )
        final_claims = tuple(claims)
        slot.validate_claims(final_claims)
        return MemoryStateTransition(
            slot,
            final_claims,
            tuple(dict.fromkeys(changed)),
            expected,
            reconciliation.relation,
        )

    def _target_state(
        self,
        profile: TransitionProfile,
        proposal: MemorySemanticProposal,
        reconciliation: ReconciliationResult,
    ) -> str:
        semantic = proposal.semantic
        assert isinstance(semantic, NormalizedSemanticAssessment)
        if reconciliation.historical_only:
            return "PROPOSED"
        if reconciliation.relation == SemanticRelation.CONTRADICTS:
            return "CONFLICTED"
        if semantic.speech_act in {SpeechAct.RETRACTION, SpeechAct.REJECTION}:
            return "RETRACTED"
        if profile in {TransitionProfile.OBSERVATIONAL, TransitionProfile.EXPERIENCE}:
            return "ACTIVE"
        confirmed = (
            semantic.speech_act in {SpeechAct.CONFIRMATION, SpeechAct.CORRECTION}
            or semantic.commitment == Commitment.CONFIRMED
        )
        if confirmed and proposal.epistemic_status == EpistemicStatus.EXPLICIT:
            return "ACTIVE"
        return "PROPOSED"

    def _next_existing_state(
        self,
        claim: MemoryClaim,
        target_state: str,
        semantic: NormalizedSemanticAssessment,
        relation: SemanticRelation,
        historical_only: bool,
    ) -> tuple[str, dict]:
        qualifiers = dict(claim.current.qualifiers)
        if historical_only:
            qualifiers["non_current_historical"] = True
            return "PROPOSED", qualifiers
        if relation == SemanticRelation.CONTRADICTS:
            return "CONFLICTED", qualifiers
        if claim.current.state == "ACTIVE" and target_state == "PROPOSED":
            return "ACTIVE", qualifiers
        qualifiers.pop("non_current_historical", None)
        if target_state == "ACTIVE":
            qualifiers.pop("phase", None)
        if semantic.speech_act == SpeechAct.EVALUATION_REQUEST and target_state == "PROPOSED":
            qualifiers["phase"] = "evaluation_candidate"
        return target_state, qualifiers

    def _proposal_qualifiers(
        self,
        semantic: NormalizedSemanticAssessment,
        target_state: str,
        historical_only: bool,
    ) -> dict:
        qualifiers: dict[str, object] = {"non_current_historical": True} if historical_only else {}
        if semantic.speech_act == SpeechAct.EVALUATION_REQUEST and target_state == "PROPOSED":
            qualifiers["phase"] = "evaluation_candidate"
        return qualifiers

    def _supersede(
        self,
        claim: MemoryClaim,
        target: MemoryClaim,
        proposal: MemorySemanticProposal,
    ) -> MemoryClaim:
        return claim.with_revision(
            MemoryRevision(
                revision=claim.latest_revision.revision + 1,
                state="SUPERSEDED",
                value_fields=claim.current.value_fields,
                evidence_refs=proposal.evidence_refs,
                proposal_id=proposal.proposal_id,
                relation=SemanticRelation.SUPERSEDES.value,
                epistemic_status=proposal.epistemic_status.value,
                field_evidence_refs=proposal.field_evidence_refs,
                proposal_fingerprint=proposal.fingerprint,
                extractor_version=proposal.extractor_version,
                model_id=proposal.model_id,
                prompt_version=proposal.prompt_version,
                policy_version=self.VERSION,
                schema_version="canonical_memory_v2",
                qualifiers={
                    "superseded_by": target.claim_id,
                    **self._provenance_qualifiers(proposal),
                },
                previous_revision=claim.latest_revision.revision,
                valid_from=self._effective_at(proposal),
            )
        )

    def _revision(
        self,
        revision: int,
        state: str,
        proposal: MemorySemanticProposal,
        relation: SemanticRelation,
        qualifiers: dict | None = None,
    ) -> MemoryRevision:
        now = utc_now()
        return MemoryRevision(
            revision=revision,
            state=state,
            value_fields=proposal.value_fields,
            evidence_refs=proposal.evidence_refs,
            proposal_id=proposal.proposal_id,
            relation=relation.value,
            epistemic_status=proposal.epistemic_status.value,
            field_evidence_refs=proposal.field_evidence_refs,
            proposal_fingerprint=proposal.fingerprint,
            extractor_version=proposal.extractor_version,
            model_id=proposal.model_id,
            prompt_version=proposal.prompt_version,
            policy_version=self.VERSION,
            schema_version="canonical_memory_v2",
            qualifiers={**(qualifiers or {}), **self._provenance_qualifiers(proposal)},
            created_at=now,
            transaction_time=now,
            previous_revision=revision - 1 if revision > 1 else None,
            valid_from=self._effective_at(proposal, fallback=now),
        )

    def _effective_at(self, proposal: MemorySemanticProposal, fallback: str | None = None) -> str:
        explicit = (
            proposal.metadata.get("effective_at")
            or proposal.metadata.get("valid_from")
            or proposal.metadata.get("occurred_at")
        )
        if explicit:
            return str(explicit)
        transition_refs = tuple(proposal.field_evidence_refs.get("transition", ()))
        occurred = sorted(str(ref.occurred_at) for ref in transition_refs if ref.occurred_at)
        return occurred[-1] if occurred else str(fallback or utc_now())

    def _provenance_qualifiers(self, proposal: MemorySemanticProposal) -> dict[str, Any]:
        qualifiers: dict[str, Any] = {
            key: str(proposal.metadata[key])
            for key in ("asserted_by", "source_role", "source_adapter_id", "source_session_id")
            if proposal.metadata.get(key)
        }
        actor_ids = tuple(
            sorted(
                {
                    str(actor_id)
                    for evidence in proposal.evidence_refs
                    if (actor_id := getattr(evidence, "actor_id", None))
                }
            )
        )
        if "asserted_by" not in qualifiers and len(actor_ids) == 1:
            qualifiers["asserted_by"] = actor_ids[0]
        elif len(actor_ids) > 1:
            qualifiers["asserted_by_multiple"] = list(actor_ids)
        qualifiers["evidence_sources"] = sorted(
            {str(evidence.source_uri) for evidence in proposal.evidence_refs if evidence.source_uri}
        )
        return qualifiers
