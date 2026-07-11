"""记忆系统里的状态转换。"""

from __future__ import annotations

from dataclasses import dataclass, replace

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
from memoryos.memory.canonical.state import (
    MemoryClaim,
    MemoryRevision,
    MemorySlot,
    TransitionProfile,
    profile_for,
)


@dataclass(frozen=True)
class MemoryStateTransition:
    """负责 MemoryStateTransition 这部分逻辑。"""

    slot: MemorySlot
    claims: tuple[MemoryClaim, ...]
    changed_claim_ids: tuple[str, ...]
    expected_revisions: dict[str, int]
    relation: SemanticRelation


class MemoryTransitionPolicy:
    """按记忆类型决定 Claim 应该怎么变更。"""

    VERSION = "memory_transition_v1"

    def apply(
        self,
        proposal: MemorySemanticProposal,
        identity: ResolvedMemoryIdentity,
        reconciliation: ReconciliationResult,
    ) -> MemoryStateTransition:
        """执行这一步处理，并保持已有状态约束。"""

        semantic = proposal.semantic
        if not isinstance(semantic, NormalizedSemanticAssessment):
            raise ValueError("transition policy accepts normalized semantic assessments only")
        profile = profile_for(proposal.memory_type)
        claims = list(reconciliation.claims)
        expected = {claim.uri: claim.current.revision for claim in claims}
        slot = reconciliation.slot or MemorySlot(
            slot_id=identity.slot_id,
            uri=identity.slot_uri,
            memory_type=proposal.memory_type,
            identity_fields=identity.slot_identity,
            scope_keys=identity.scope_keys,
        )
        expected[slot.uri] = slot.revision
        target = reconciliation.claim
        target_state = self._target_state(profile, proposal)
        changed: list[str] = []

        if target is None:
            if reconciliation.relation == SemanticRelation.CONTRADICTS and target_state != "ACTIVE":
                target_state = "CONFLICTED"
            revision = self._revision(
                1,
                target_state,
                proposal,
                reconciliation.relation,
                self._proposal_qualifiers(semantic, target_state),
            )
            target = MemoryClaim(
                identity.claim_id,
                identity.claim_uri,
                identity.slot_id,
                identity.canonical_value,
                profile,
                (revision,),
            )
            claims.append(target)
            expected[target.uri] = 0
            changed.append(target.claim_id)
        else:
            next_state, qualifiers = self._next_existing_state(target, target_state, semantic, reconciliation.relation)
            if (
                next_state != target.current.state
                or qualifiers != dict(target.current.qualifiers)
                or dict(target.current.value_fields) != dict(proposal.value_fields)
            ):
                target = target.with_revision(
                    self._revision(
                        target.current.revision + 1,
                        next_state,
                        proposal,
                        reconciliation.relation,
                        qualifiers,
                    )
                )
                claims = [target if item.claim_id == target.claim_id else item for item in claims]
                changed.append(target.claim_id)

        active_claim_id = slot.active_claim_id
        if target.current.state == "ACTIVE" and profile == TransitionProfile.AUTHORITATIVE_STATE:
            for index, claim in enumerate(claims):
                if claim.claim_id == target.claim_id or claim.current.state != "ACTIVE":
                    continue
                claims[index] = claim.with_revision(
                    MemoryRevision(
                        revision=claim.current.revision + 1,
                        state="SUPERSEDED",
                        value_fields=claim.current.value_fields,
                        evidence_refs=proposal.evidence_refs,
                        proposal_id=proposal.proposal_id,
                        relation=SemanticRelation.SUPERSEDES.value,
                        epistemic_status=proposal.epistemic_status.value,
                        proposal_fingerprint=proposal.fingerprint,
                        extractor_version=proposal.extractor_version,
                        model_id=proposal.model_id,
                        prompt_version=proposal.prompt_version,
                        policy_version=self.VERSION,
                        schema_version="canonical_memory_v1",
                        qualifiers={"superseded_by": target.claim_id},
                        previous_revision=claim.current.revision,
                    )
                )
                changed.append(claim.claim_id)
            active_claim_id = target.claim_id
        elif target.current.state == "ACTIVE":
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
        active = [claim for claim in claims if claim.current.state == "ACTIVE"]
        if profile == TransitionProfile.AUTHORITATIVE_STATE and len(active) > 1:
            raise ValueError("authoritative slot cannot contain more than one ACTIVE claim")
        return MemoryStateTransition(
            slot,
            tuple(claims),
            tuple(dict.fromkeys(changed)),
            expected,
            reconciliation.relation,
        )

    def _target_state(self, profile: TransitionProfile, proposal: MemorySemanticProposal) -> str:
        semantic = proposal.semantic
        assert isinstance(semantic, NormalizedSemanticAssessment)
        if profile == TransitionProfile.OBSERVATIONAL:
            return "ACTIVE"
        if profile == TransitionProfile.EXPERIENCE:
            return "ACTIVE"
        if semantic.speech_act in {SpeechAct.RETRACTION, SpeechAct.REJECTION}:
            return "RETRACTED"
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
    ) -> tuple[str, dict]:
        if relation == SemanticRelation.CONTRADICTS and target_state != "ACTIVE":
            return "CONFLICTED", dict(claim.current.qualifiers)
        if claim.current.state == "ACTIVE" and target_state == "PROPOSED":
            return "ACTIVE", dict(claim.current.qualifiers)
        qualifiers = dict(claim.current.qualifiers)
        if target_state == "ACTIVE":
            qualifiers.pop("phase", None)
        if semantic.speech_act == SpeechAct.EVALUATION_REQUEST and target_state == "PROPOSED":
            qualifiers["phase"] = "evaluation_candidate"
        return target_state, qualifiers

    def _proposal_qualifiers(self, semantic: NormalizedSemanticAssessment, target_state: str) -> dict:
        if semantic.speech_act == SpeechAct.EVALUATION_REQUEST and target_state == "PROPOSED":
            return {"phase": "evaluation_candidate"}
        return {}

    def _revision(
        self,
        revision: int,
        state: str,
        proposal: MemorySemanticProposal,
        relation: SemanticRelation,
        qualifiers: dict | None = None,
    ) -> MemoryRevision:
        return MemoryRevision(
            revision=revision,
            state=state,
            value_fields=proposal.value_fields,
            evidence_refs=proposal.evidence_refs,
            proposal_id=proposal.proposal_id,
            relation=relation.value,
            epistemic_status=proposal.epistemic_status.value,
            proposal_fingerprint=proposal.fingerprint,
            extractor_version=proposal.extractor_version,
            model_id=proposal.model_id,
            prompt_version=proposal.prompt_version,
            policy_version=self.VERSION,
            schema_version="canonical_memory_v1",
            qualifiers=qualifiers or {},
            previous_revision=revision - 1 if revision > 1 else None,
        )
