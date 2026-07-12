"""The sole canonical Claim state transition policy."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, NoReturn

from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.core.time import utc_now
from memoryos.memory.canonical.identity import ResolvedMemoryIdentity, canonical_identity_value
from memoryos.memory.canonical.proposal import (
    Commitment,
    EpistemicStatus,
    MemorySemanticProposal,
    NormalizedSemanticAssessment,
    PendingMemoryProposal,
    SemanticRelation,
    SpeechAct,
    TemporalScope,
)
from memoryos.memory.canonical.reconcile import ReconciliationResult
from memoryos.memory.canonical.semantic import EligibilityDisposition, MemoryTypeEligibilityPolicy
from memoryos.memory.canonical.state import MemoryClaim, MemoryRevision, MemorySlot, TransitionProfile, profile_for
from memoryos.memory.schema import MemoryType, MemoryTypeRegistry, MemoryTypeSchema

_REPLACEMENT_RELATIONS = {SemanticRelation.CORRECTS, SemanticRelation.SUPERSEDES}
_AUTHORITATIVE_SOURCE_ROLES = frozenset({"user", "system"})
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


class PendingSemanticReconciliation(ValueError):
    """Fail closed without producing a canonical state change."""

    def __init__(self, relation: SemanticRelation, reason: str) -> None:
        self.relation = relation
        self.reason = reason
        super().__init__(f"semantic reconciliation pending: {relation.value}:{reason}")


class _DestructiveEffectAuthority(str, Enum):
    STRUCTURED_PENDING_REVIEW = "STRUCTURED_PENDING_REVIEW"
    STRUCTURED_EXPLICIT_COMMAND = "STRUCTURED_EXPLICIT_COMMAND"


class _DestructiveEffectAuthorization:
    """Opaque, one-use capability registered by exactly one transition policy."""

    __slots__ = ("__weakref__",)

    def __copy__(self) -> NoReturn:
        raise TypeError("destructive effect authorization cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("destructive effect authorization cannot be copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("destructive effect authorization cannot be serialized")


@dataclass(frozen=True)
class _DestructiveEffectBinding:
    authority: _DestructiveEffectAuthority
    authorization_id: str
    proposal_fingerprint: str
    target_claim_ids: tuple[str, ...]
    owner_user_id: str
    tenant_id: str
    pending_uri: str = ""
    pending_lifecycle_revision: int = 0


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

    VERSION = "memory_transition_v3"

    def __init__(
        self,
        registry: MemoryTypeRegistry | None = None,
        eligibility_policy: MemoryTypeEligibilityPolicy | None = None,
    ) -> None:
        self.registry = registry or MemoryTypeRegistry()
        self.eligibility_policy = eligibility_policy or MemoryTypeEligibilityPolicy()
        self.__effect_authorizations: dict[_DestructiveEffectAuthorization, _DestructiveEffectBinding] = {}

    def __issue_effect_authorization(
        self,
        *,
        authority: _DestructiveEffectAuthority,
        authorization_id: str,
        proposal: MemorySemanticProposal,
        target_claim_ids: tuple[str, ...],
        owner_user_id: str,
        tenant_id: str,
        pending_uri: str = "",
        pending_lifecycle_revision: int = 0,
    ) -> _DestructiveEffectAuthorization:
        """Issue an in-process capability bound to this transition-policy instance."""

        normalized_targets = tuple(sorted({str(item) for item in target_claim_ids if item}))
        if not authorization_id or not proposal.fingerprint:
            raise ValueError("destructive effect authorization requires an id and proposal fingerprint")
        if not owner_user_id or not tenant_id:
            raise ValueError("destructive effect authorization requires owner and tenant")
        if authority == _DestructiveEffectAuthority.STRUCTURED_PENDING_REVIEW:
            if not pending_uri or pending_lifecycle_revision < 1:
                raise ValueError("structured pending review authorization requires pending lifecycle identity")
        elif authority == _DestructiveEffectAuthority.STRUCTURED_EXPLICIT_COMMAND:
            self._validate_explicit_command_authority(proposal, normalized_targets)
        authorization = _DestructiveEffectAuthorization()
        self.__effect_authorizations[authorization] = _DestructiveEffectBinding(
            authority=authority,
            authorization_id=authorization_id,
            proposal_fingerprint=proposal.fingerprint,
            target_claim_ids=normalized_targets,
            owner_user_id=owner_user_id,
            tenant_id=tenant_id,
            pending_uri=pending_uri,
            pending_lifecycle_revision=pending_lifecycle_revision,
        )
        return authorization

    def apply(
        self,
        proposal: MemorySemanticProposal,
        identity: ResolvedMemoryIdentity,
        reconciliation: ReconciliationResult,
    ) -> MemoryStateTransition:
        """Apply only non-destructive transitions through the public policy surface."""

        return self._apply(
            proposal,
            identity,
            reconciliation,
            effect_authorization=None,
        )

    def _apply_confirmed_pending_review(
        self,
        pending: PendingMemoryProposal,
        proposal: MemorySemanticProposal,
        identity: ResolvedMemoryIdentity,
        reconciliation: ReconciliationResult,
        *,
        authorization_id: str,
        owner_user_id: str,
        tenant_id: str,
    ) -> MemoryStateTransition:
        """Apply a destructive effect only from one durable CONFIRMED pending record."""

        if (
            pending.lifecycle_state != LifecycleState.CONFIRMED
            or not pending.uri.startswith(f"memoryos://user/{owner_user_id}/memories/pending/")
            or pending.scope.visibility.tenant_id != tenant_id
            or pending.proposal.memory_type != proposal.memory_type
            or dict(pending.proposal.identity_fields) != dict(proposal.identity_fields)
            or dict(pending.proposal.value_fields) != dict(proposal.value_fields)
            or pending.proposal.related_memory_ids != proposal.related_memory_ids
            or pending.proposal.related_slot_ids != proposal.related_slot_ids
            or pending.proposal.related_claim_ids != proposal.related_claim_ids
        ):
            raise PendingSemanticReconciliation(
                reconciliation.relation,
                "destructive_effect_requires_confirmed_pending_record",
            )
        active = reconciliation.active_claim
        authorization = self.__issue_effect_authorization(
            authority=_DestructiveEffectAuthority.STRUCTURED_PENDING_REVIEW,
            authorization_id=authorization_id,
            proposal=proposal,
            target_claim_ids=(active.claim_id,) if active is not None else (),
            owner_user_id=owner_user_id,
            tenant_id=tenant_id,
            pending_uri=pending.uri,
            pending_lifecycle_revision=pending.lifecycle_revision,
        )

        return self._apply(
            proposal,
            identity,
            reconciliation,
            effect_authorization=authorization,
        )

    def _apply_structured_retraction(
        self,
        proposal: MemorySemanticProposal,
        identity: ResolvedMemoryIdentity,
        reconciliation: ReconciliationResult,
        *,
        authorization_id: str,
        owner_user_id: str,
        tenant_id: str,
    ) -> MemoryStateTransition:
        """Apply an exact structured retraction command after proof validation."""

        active = reconciliation.active_claim
        authorization = self.__issue_effect_authorization(
            authority=_DestructiveEffectAuthority.STRUCTURED_EXPLICIT_COMMAND,
            authorization_id=authorization_id,
            proposal=proposal,
            target_claim_ids=(active.claim_id,) if active is not None else (),
            owner_user_id=owner_user_id,
            tenant_id=tenant_id,
        )
        return self._apply(
            proposal,
            identity,
            reconciliation,
            effect_authorization=authorization,
        )

    def _apply(
        self,
        proposal: MemorySemanticProposal,
        identity: ResolvedMemoryIdentity,
        reconciliation: ReconciliationResult,
        *,
        effect_authorization: _DestructiveEffectAuthorization | None,
    ) -> MemoryStateTransition:
        semantic = proposal.semantic
        if not isinstance(semantic, NormalizedSemanticAssessment):
            raise PendingSemanticReconciliation(SemanticRelation.SCHEMA_MISMATCH, "semantic_not_normalized")
        if not reconciliation.transition_allowed:
            raise PendingSemanticReconciliation(reconciliation.relation, reconciliation.pending_reason)
        if reconciliation.relation in {SemanticRelation.ALTERNATIVE, SemanticRelation.CONTRADICTS}:
            raise PendingSemanticReconciliation(
                reconciliation.relation,
                "nonfinal_relation_requires_review",
            )
        self._validate_v3_effect_gate(proposal, reconciliation)
        self._validate_destructive_effect_authorization(
            proposal,
            reconciliation,
            authorization=effect_authorization,
        )
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
        self._validate_retraction(proposal, reconciliation, target)
        self._validate_replacement(proposal, identity, reconciliation)
        self._guard_unconfirmed_active_supplement(proposal, reconciliation, target)
        changed: list[str] = []

        if target is None:
            revision = self._revision(
                1,
                target_state,
                proposal,
                reconciliation.relation,
                self._proposal_qualifiers(proposal, semantic, target_state, reconciliation.historical_only),
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
                proposal,
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
                if reconciliation.relation not in _REPLACEMENT_RELATIONS:
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

    def _validate_v3_effect_gate(
        self,
        proposal: MemorySemanticProposal,
        reconciliation: ReconciliationResult,
    ) -> None:
        """Repeat the V3 admission invariants at the final effect boundary."""

        if str(getattr(proposal, "semantic_contract_version", "v2")).casefold() != "v3":
            raise PendingSemanticReconciliation(
                reconciliation.relation,
                "semantic_contract_v3_required",
            )
        semantic = proposal.semantic
        assert isinstance(semantic, NormalizedSemanticAssessment)

        try:
            memory_type = MemoryType(proposal.memory_type)
            schema = self.registry.get(memory_type)
        except ValueError as exc:
            raise PendingSemanticReconciliation(
                reconciliation.relation,
                "unsupported_memory_schema",
            ) from exc

        utterance = self._semantic_value(semantic, "utterance_mode")
        attribution = self._semantic_value(semantic, "attribution")
        durability = self._semantic_value(semantic, "durability")
        atomicity = self._semantic_value(semantic, "atomicity")
        if not semantic.schema_safe:
            raise PendingSemanticReconciliation(
                reconciliation.relation,
                "semantic_v3_unknown_or_mixed_or_compound_or_third_party",
            )
        if (
            utterance in {"QUESTION", "HYPOTHETICAL"}
            or attribution == "QUOTED"
            or durability == "TRANSIENT"
        ):
            raise PendingSemanticReconciliation(
                reconciliation.relation,
                "semantic_v3_question_or_hypothetical_or_quoted_or_transient",
            )
        if (
            utterance not in {"ASSERTION", "DIRECTIVE"}
            or attribution != "SOURCE_ACTOR"
            or durability != "DURABLE"
            or atomicity != "ATOMIC"
        ):
            raise PendingSemanticReconciliation(
                reconciliation.relation,
                "semantic_v3_unknown_or_mixed_or_compound_or_third_party",
            )
        if semantic.speech_act in {SpeechAct.PROPOSAL, SpeechAct.EVALUATION_REQUEST}:
            raise PendingSemanticReconciliation(
                reconciliation.relation,
                "nonfinal_relation_requires_review",
            )
        atomic_ref = getattr(proposal, "atomic_evidence_ref", None)
        atomic_bindings = (
            "transition",
            "semantic.speech_act",
            "semantic.commitment",
            "semantic.temporal_scope",
            "semantic.relation_to_existing",
            "semantic.utterance_mode",
            "semantic.attribution",
            "semantic.durability",
            "semantic.modal_force",
            "semantic.atomicity",
        )
        if (
            atomic_ref is None
            or atomic_ref not in proposal.evidence_refs
            or getattr(atomic_ref, "span_start", None) is None
            or getattr(atomic_ref, "span_end", None) is None
            or proposal.metadata.get("semantic_contract_validated") is not True
            or proposal.metadata.get("atomic_evidence_validated") is not True
            or proposal.metadata.get("transition_evidence_validated") is not True
            or any(tuple(proposal.field_evidence_refs.get(field_name, ())) != (atomic_ref,) for field_name in atomic_bindings)
        ):
            raise PendingSemanticReconciliation(
                reconciliation.relation,
                "atomic_evidence_invalid_or_missing",
            )
        source_role = str(getattr(atomic_ref, "actor_kind", "") or "").strip().casefold()
        eligibility = self.eligibility_policy.evaluate(
            proposal,
            memory_type=memory_type,
            schema=schema,
            source_role=source_role,
        )
        if eligibility.disposition != EligibilityDisposition.ELIGIBLE:
            raise PendingSemanticReconciliation(
                reconciliation.relation,
                eligibility.reason,
            )

    def _validate_destructive_effect_authorization(
        self,
        proposal: MemorySemanticProposal,
        reconciliation: ReconciliationResult,
        *,
        authorization: _DestructiveEffectAuthorization | None,
    ) -> None:
        semantic = proposal.semantic
        assert isinstance(semantic, NormalizedSemanticAssessment)
        destructive = (
            reconciliation.relation in _REPLACEMENT_RELATIONS
            or semantic.relation_to_existing in _REPLACEMENT_RELATIONS
            or semantic.speech_act in {SpeechAct.RETRACTION, SpeechAct.REJECTION}
        )
        if not destructive:
            return
        active = reconciliation.active_claim
        target_claim_ids = tuple(sorted({active.claim_id} if active is not None else set()))
        atomic_ref = proposal.atomic_evidence_ref
        actor_kind = str(getattr(atomic_ref, "actor_kind", "") or "").casefold()
        actor_id = str(getattr(atomic_ref, "actor_id", "") or "")
        evidence_tenant = str(getattr(atomic_ref, "tenant_id", "") or "default")
        binding = (
            self.__effect_authorizations.pop(authorization, None)
            if isinstance(authorization, _DestructiveEffectAuthorization)
            else None
        )
        if (
            binding is None
            or binding.proposal_fingerprint != proposal.fingerprint
            or binding.target_claim_ids != target_claim_ids
            or binding.tenant_id != evidence_tenant
            or (actor_kind == "user" and binding.owner_user_id != actor_id)
        ):
            raise PendingSemanticReconciliation(
                reconciliation.relation,
                "destructive_effect_requires_structured_review",
            )

    def _validate_explicit_command_authority(
        self,
        proposal: MemorySemanticProposal,
        target_claim_ids: tuple[str, ...],
    ) -> None:
        semantic = proposal.semantic
        if (
            not isinstance(semantic, NormalizedSemanticAssessment)
            or semantic.speech_act not in {SpeechAct.RETRACTION, SpeechAct.REJECTION}
            or proposal.metadata.get("effect_authority") != "structured_explicit_command"
            or len(target_claim_ids) != 1
            or proposal.atomic_evidence_ref is None
            or not proposal.atomic_evidence_ref.quoted_text
        ):
            raise ValueError("structured explicit command authority requires an exact retraction command")
        try:
            command = json.loads(proposal.atomic_evidence_ref.quoted_text)
        except (TypeError, ValueError) as exc:
            raise ValueError("structured explicit command evidence must be JSON") from exc
        if (
            not isinstance(command, dict)
            or command.get("command") != "RETRACT_CANONICAL_CLAIM"
            or str(command.get("claim_id") or "") != target_claim_ids[0]
            or str(command.get("memory_type") or "") != proposal.memory_type
        ):
            raise ValueError("structured explicit command does not match the retraction target")

    def _semantic_value(self, semantic: NormalizedSemanticAssessment, field_name: str) -> str:
        value = getattr(semantic, field_name, None)
        return str(getattr(value, "value", value) or "UNKNOWN").strip().upper()

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
        if reconciliation.relation == SemanticRelation.ALTERNATIVE:
            return "PROPOSED"
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
        proposal: MemorySemanticProposal,
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
        if relation == SemanticRelation.SUPPLEMENTS:
            display_fields = self._display_fields(proposal)
            if display_fields:
                qualifiers["display_fields"] = {
                    **dict(qualifiers.get("display_fields", {}) or {}),
                    **display_fields,
                }
        return target_state, qualifiers

    def _proposal_qualifiers(
        self,
        proposal: MemorySemanticProposal,
        semantic: NormalizedSemanticAssessment,
        target_state: str,
        historical_only: bool,
    ) -> dict:
        qualifiers: dict[str, object] = {"non_current_historical": True} if historical_only else {}
        if semantic.speech_act == SpeechAct.EVALUATION_REQUEST and target_state == "PROPOSED":
            qualifiers["phase"] = "evaluation_candidate"
        display_fields = self._display_fields(proposal)
        if display_fields:
            qualifiers["display_fields"] = display_fields
        return qualifiers

    def _validate_replacement(
        self,
        proposal: MemorySemanticProposal,
        identity: ResolvedMemoryIdentity,
        reconciliation: ReconciliationResult,
    ) -> None:
        relation = reconciliation.relation
        if relation not in _REPLACEMENT_RELATIONS:
            return
        semantic = proposal.semantic
        assert isinstance(semantic, NormalizedSemanticAssessment)
        if semantic.speech_act in {SpeechAct.RETRACTION, SpeechAct.REJECTION}:
            return
        if reconciliation.active_claim is not None and identity.claim_id == reconciliation.active_claim.claim_id:
            raise PendingSemanticReconciliation(
                SemanticRelation.AMBIGUOUS,
                "replacement_cannot_target_same_claim",
            )
        if semantic.relation_to_existing != relation:
            raise PendingSemanticReconciliation(
                SemanticRelation.AMBIGUOUS,
                "replacement_cannot_upgrade_from_non_replacement_relation",
            )
        active = reconciliation.active_claim
        if active is None or not self._explicitly_targets_active(proposal, active):
            raise PendingSemanticReconciliation(
                SemanticRelation.AMBIGUOUS,
                "replacement_requires_explicit_active_claim_target",
            )
        if semantic.temporal_scope != TemporalScope.CURRENT:
            raise PendingSemanticReconciliation(
                SemanticRelation.AMBIGUOUS,
                "replacement_requires_current_temporality",
            )
        if self._applicability_conflicts(proposal, active):
            raise PendingSemanticReconciliation(
                SemanticRelation.AMBIGUOUS,
                "replacement_applicability_conflicts_with_active_claim",
            )
        transition_refs = tuple(proposal.field_evidence_refs.get("transition", ()))
        relation_refs = tuple(proposal.field_evidence_refs.get("semantic.relation_to_existing", ()))
        if not (
            proposal.epistemic_status == EpistemicStatus.EXPLICIT
            and proposal.metadata.get("transition_evidence_validated") is True
            and proposal.metadata.get("semantic_relation_evidence_validated") is True
            and proposal.metadata.get("replacement_evidence_validated") is True
            and transition_refs
            and relation_refs
            and set(transition_refs).issubset(proposal.evidence_refs)
            and set(relation_refs).issubset(proposal.evidence_refs)
            and self._source_allowed(proposal, transition_refs)
        ):
            raise PendingSemanticReconciliation(
                SemanticRelation.AMBIGUOUS,
                "replacement_requires_validated_authoritative_evidence",
            )

    def _validate_retraction(
        self,
        proposal: MemorySemanticProposal,
        reconciliation: ReconciliationResult,
        target: MemoryClaim | None,
    ) -> None:
        semantic = proposal.semantic
        assert isinstance(semantic, NormalizedSemanticAssessment)
        if semantic.speech_act not in {SpeechAct.RETRACTION, SpeechAct.REJECTION}:
            return
        active = reconciliation.active_claim
        transition_refs = tuple(proposal.field_evidence_refs.get("transition", ()))
        if (
            semantic.relation_to_existing != SemanticRelation.CORRECTS
            or active is None
            or target is None
            or target.claim_id != active.claim_id
            or not self._explicitly_targets_active(proposal, active)
            or proposal.epistemic_status != EpistemicStatus.EXPLICIT
            or not transition_refs
            or not set(transition_refs).issubset(proposal.evidence_refs)
            or not self._source_allowed(proposal, transition_refs)
        ):
            raise PendingSemanticReconciliation(
                SemanticRelation.AMBIGUOUS,
                "retraction_requires_explicit_authoritative_active_claim_target",
            )

    def _guard_unconfirmed_active_supplement(
        self,
        proposal: MemorySemanticProposal,
        reconciliation: ReconciliationResult,
        target: MemoryClaim | None,
    ) -> None:
        active = reconciliation.active_claim
        display_fields = self._display_fields(proposal)
        current_display_fields = dict(target.current.qualifiers.get("display_fields", {}) or {}) if target else {}
        core_changed = target is not None and dict(target.current.value_fields) != dict(proposal.value_fields)
        display_changed = bool(display_fields) and display_fields != current_display_fields
        if (
            active is None
            or target is None
            or target.claim_id != active.claim_id
            or not (core_changed or display_changed)
            or reconciliation.relation not in {SemanticRelation.DUPLICATE, SemanticRelation.SUPPLEMENTS}
        ):
            return
        semantic = proposal.semantic
        assert isinstance(semantic, NormalizedSemanticAssessment)
        confirmed = (
            proposal.epistemic_status == EpistemicStatus.EXPLICIT
            and proposal.metadata.get("transition_evidence_validated") is True
            and (
                semantic.speech_act in {SpeechAct.CONFIRMATION, SpeechAct.CORRECTION}
                or semantic.commitment == Commitment.CONFIRMED
            )
            and self._source_allowed(
                proposal,
                tuple(proposal.field_evidence_refs.get("transition", ())),
            )
        )
        if not confirmed:
            raise PendingSemanticReconciliation(
                SemanticRelation.SUPPLEMENTS,
                "unconfirmed_supplement_cannot_revise_active_claim",
            )

    def _display_fields(self, proposal: MemorySemanticProposal) -> dict[str, object]:
        raw = proposal.metadata.get("display_fields", {})
        return {str(key): value for key, value in dict(raw or {}).items()} if isinstance(raw, Mapping) else {}

    def _explicitly_targets_active(self, proposal: MemorySemanticProposal, active: MemoryClaim) -> bool:
        explicit_claim_targets = {
            *proposal.related_claim_ids,
            *proposal.related_memory_ids,
        }
        return active.claim_id in explicit_claim_targets or active.uri in explicit_claim_targets

    def _source_allowed(
        self,
        proposal: MemorySemanticProposal,
        refs: tuple[object, ...],
        schema: MemoryTypeSchema | None = None,
    ) -> bool:
        try:
            effective_schema = schema or self.registry.get(proposal.memory_type)
        except ValueError:
            return False
        declared = str(proposal.metadata.get("source_role") or "").strip().casefold()
        actor_kinds = {
            str(actor_kind).strip().casefold()
            for ref in refs
            if (actor_kind := getattr(ref, "actor_kind", None))
        }
        if declared and actor_kinds and declared not in actor_kinds:
            return False
        roles = actor_kinds or ({declared} if declared else set())
        if not roles:
            return False

        def allowed(role: str) -> bool:
            if role == "system":
                return True
            if role == "user":
                return effective_schema.allow_user_source
            if role in {"assistant", "agent"}:
                return effective_schema.allow_assistant_source
            if role == "tool":
                return effective_schema.allow_tool_source
            return False

        return all(allowed(role) for role in roles)

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

    def _present(self, value: object) -> bool:
        return value is not None and value != "" and value != () and value != [] and value != {}

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
