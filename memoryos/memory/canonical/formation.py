"""记忆系统里的形成。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.source_store import RelationStore, SourceStore
from memoryos.core.ids import stable_hash
from memoryos.memory.canonical.admission import (
    ProposalAdmissionDecision,
    ProposalAdmissionGate,
)
from memoryos.memory.canonical.episode import EvidenceEpisode
from memoryos.memory.canonical.event import canonicalize
from memoryos.memory.canonical.evidence import EvidenceRef, ProposalEvidenceValidator, bind_field_evidence
from memoryos.memory.canonical.identity import (
    IDENTITY_ALGORITHM_V2,
    AliasRegistry,
    ResolvedMemoryIdentity,
    StableMemoryIdentityResolver,
)
from memoryos.memory.canonical.proposal import (
    EpistemicStatus,
    MemorySemanticProposal,
    NormalizedSemanticAssessment,
    PendingMemoryProposal,
    SemanticAssessment,
    SemanticRelation,
)
from memoryos.memory.canonical.reconcile import MemorySemanticReconciler
from memoryos.memory.canonical.repository import CanonicalMemoryRepository
from memoryos.memory.canonical.scope import (
    AuthorityPolicy,
    MemoryScope,
    ScopeRef,
    ScopeResolutionSource,
    ScopeSelector,
    VisibilityPolicy,
    scope_from_external,
)
from memoryos.memory.canonical.semantic import MemorySemanticNormalizer, MemoryTypeEligibilityPolicy
from memoryos.memory.canonical.state import (
    MemoryClaim,
    MemorySlot,
    materialized_current_revision_payload,
)
from memoryos.memory.canonical.transaction import MemoryTransactionPlanner
from memoryos.memory.canonical.transition import (
    MemoryTransitionPolicy,
    PendingSemanticReconciliation,
)
from memoryos.memory.canonical.visibility import read_committed_pending
from memoryos.memory.schema import MemoryCandidateDraft, MemoryType, MemoryTypeRegistry
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


@dataclass(frozen=True)
class CanonicalFormationResult:
    """保存 CanonicalFormationResult 需要的这组数据。"""

    operations: tuple[ContextOperation, ...]
    decision: ProposalAdmissionDecision
    reason: str
    proposal: MemorySemanticProposal
    pending_uri: str = ""
    pending_lifecycle_state: str = ""
    pending_lifecycle_revision: int = 0
    pending_existing: bool = False
    resolved_identity: ResolvedMemoryIdentity | None = None


class CandidateProposalAdapter:
    """Convert a typed candidate draft into an evidence-bound proposal."""

    FALLBACK_PENDING_REASON = "PENDING_FALLBACK_REQUIRES_SEMANTIC_REVIEW"

    def adapt(
        self,
        candidate: MemoryCandidateDraft,
        episode: EvidenceEpisode,
        archive: SessionArchive,
    ) -> MemorySemanticProposal:
        """处理 adapt 这一步。"""

        events = [episode.event(event_id) for event_id in candidate.source_message_ids]
        matched = [event for event in events if event is not None]
        base_refs = tuple(EvidenceRef.from_event(event, source_uri=archive.archive_uri) for event in matched)
        identity, system_fields = self._identity(candidate, episode)
        identity_resolution_pending = candidate.memory_type == MemoryType.PROFILE and not candidate.fields.get(
            "attribute_key"
        )
        speech, commitment, temporal, relation = self._semantic(candidate)
        epistemic = self._epistemic(candidate.source_role)
        value_fields = self._values(candidate)
        evidence_refs, field_bindings = self._field_evidence(
            identity,
            value_fields,
            system_fields,
            candidate.content,
            matched,
            base_refs,
            archive.archive_uri,
        )
        return MemorySemanticProposal(
            proposal_id=f"proposal_{stable_hash([archive.task_id, candidate.memory_type.value, identity, candidate.fields, candidate.source_message_ids], length=32)}",
            memory_type=candidate.memory_type.value,
            identity_fields=identity,
            value_fields=value_fields,
            semantic=SemanticAssessment(speech, commitment, temporal, relation),
            epistemic_status=epistemic,
            suggested_scope_refs=self._suggested_scopes(candidate, episode, archive.user_id),
            related_memory_ids=tuple(str(item) for item in candidate.fields.get("_related_memory_ids", []) or []),
            related_slot_ids=tuple(str(item) for item in candidate.fields.get("_related_slot_ids", []) or []),
            related_claim_ids=tuple(str(item) for item in candidate.fields.get("_related_claim_ids", []) or []),
            evidence_refs=evidence_refs,
            field_evidence_refs=bind_field_evidence(
                identity,
                value_fields,
                evidence_refs,
                bindings=field_bindings,
            ),
            confidence=candidate.confidence,
            extractor_version="candidate_proposal_adapter_v2",
            model_id=None,
            metadata={
                "source_role": candidate.source_role,
                "source_adapter_id": candidate.source_adapter_id,
                "source_session_id": candidate.source_session_id or archive.session_id,
                "source_connect": dict(archive.metadata.get("connect", {}) or {}),
                "system_identity_fields": system_fields,
                "candidate_reason": candidate.reason,
                "replacement_explicit": bool(candidate.fields.get("_replacement_explicit", False)),
                "identity_resolution_pending": identity_resolution_pending,
                # MemoryCandidateDraft is the conservative fallback/legacy
                # carrier.  Its rule-derived semantic hints are useful for
                # review, but can never grant canonical write authority.
                "fallback_pending_only": True,
            },
        )

    def _identity(self, candidate: MemoryCandidateDraft, episode: EvidenceEpisode) -> tuple[dict[str, Any], list[str]]:
        fields = candidate.fields
        topic = self._topic(candidate.content)
        if candidate.memory_type == MemoryType.PROFILE:
            explicit_key = fields.get("attribute_key")
            return ({"attribute_key": str(explicit_key)}, ["attribute_key"]) if explicit_key else ({}, [])
        if candidate.memory_type == MemoryType.PREFERENCE:
            subject = str(fields.get("subject") or topic)
            dimension = str(fields.get("dimension") or topic)
            system_fields = [field for field in ("subject", "dimension") if fields.get(field)]
            return {"subject": subject, "dimension": dimension}, system_fields
        if candidate.memory_type == MemoryType.ENTITY:
            explicit_entity = fields.get("canonical_entity_id") or fields.get("name")
            return {
                "entity_type": str(fields.get("entity_type") or fields.get("type") or "entity"),
                "canonical_entity_id": str(explicit_entity or topic),
            }, ["entity_type", "canonical_entity_id"] if explicit_entity else []
        if candidate.memory_type == MemoryType.PROJECT_RULE:
            explicit_topic = fields.get("rule_topic") or fields.get("rule_key")
            return {"rule_topic": str(explicit_topic or topic)}, ["rule_topic"] if explicit_topic else []
        if candidate.memory_type == MemoryType.PROJECT_DECISION:
            explicit_topic = fields.get("decision_topic") or fields.get("decision_key")
            return {"decision_topic": str(explicit_topic or topic)}, ["decision_topic"] if explicit_topic else []
        if candidate.memory_type == MemoryType.EVENT:
            source_event_id = candidate.source_message_ids[0] if candidate.source_message_ids else topic
            event_key = f"{episode.episode_id}:{source_event_id}"
            return {"event_key": event_key}, ["event_key"]
        return {
            "task_pattern": str(fields.get("task_pattern") or topic),
            "environment_signature": str(
                fields.get("environment_signature") or episode.origin.primary_scope.key
                if episode.origin.primary_scope
                else episode.origin.adapter_id
            ),
        }, [
            field_name
            for field_name in ("task_pattern", "environment_signature")
            if fields.get(field_name) or field_name == "environment_signature"
        ]

    def _values(self, candidate: MemoryCandidateDraft) -> dict[str, Any]:
        key = {
            MemoryType.PROFILE: "summary",
            MemoryType.PREFERENCE: "preference",
            MemoryType.ENTITY: "name",
            MemoryType.EVENT: "event",
            MemoryType.PROJECT_RULE: "rule",
            MemoryType.PROJECT_DECISION: "decision",
            MemoryType.AGENT_EXPERIENCE: "outcome",
        }[candidate.memory_type]
        semantic_fields = {
            field_name: value
            for field_name, value in candidate.fields.items()
            if not field_name.startswith("_")
            and field_name
            not in {
                "project_id",
                "adapter_id",
                "tenant_id",
                "scope",
                "attribute_key",
                "subject",
                "dimension",
                "canonical_entity_id",
                "rule_topic",
                "rule_key",
                "decision_topic",
                "decision_key",
                "event_key",
                "task_pattern",
                "environment_signature",
            }
        }
        values = {key: candidate.content, **semantic_fields}
        if candidate.memory_type == MemoryType.PROFILE and "canonical_value" not in values:
            values["canonical_value"] = candidate.content
        elif candidate.memory_type == MemoryType.PREFERENCE and "canonical_value" not in values:
            values["canonical_value"] = str(
                candidate.fields.get("preference_value") or candidate.fields.get("preference") or candidate.content
            )
        elif candidate.memory_type == MemoryType.ENTITY and "canonical_value" not in values:
            values["canonical_value"] = str(candidate.fields.get("name") or candidate.content)
        elif candidate.memory_type == MemoryType.EVENT and "canonical_value" not in values:
            values["canonical_value"] = str(candidate.fields.get("event") or candidate.content)
        elif candidate.memory_type == MemoryType.AGENT_EXPERIENCE and "canonical_value" not in values:
            values["canonical_value"] = str(candidate.fields.get("outcome") or candidate.content)
        return values

    def _field_evidence(
        self,
        identity_fields: Mapping[str, Any],
        value_fields: Mapping[str, Any],
        system_identity_fields: list[str],
        candidate_content: str,
        events: list[Any],
        base_refs: tuple[EvidenceRef, ...],
        source_uri: str,
    ) -> tuple[tuple[EvidenceRef, ...], dict[str, tuple[EvidenceRef, ...]]]:
        """Bind each semantic field to the exact immutable source span when possible."""

        all_refs: list[EvidenceRef] = list(base_refs)
        bindings: dict[str, tuple[EvidenceRef, ...]] = {}
        system_fields = set(system_identity_fields)
        for prefix, fields in (("identity", identity_fields), ("value", value_fields)):
            for key, value in fields.items():
                field_name = f"{prefix}.{key}"
                if prefix == "identity" and key in system_fields:
                    bindings[field_name] = base_refs
                    continue
                refs = self._value_refs(value, events, source_uri)
                if (
                    not refs
                    and prefix == "value"
                    and key in {"canonical_value", "polarity", "constraint_polarity"}
                    and str(value).casefold()
                    in {
                        "forbidden",
                        "required",
                        "allowed",
                        "preferred",
                        "discouraged",
                        "require",
                        "forbid",
                        "allow",
                        "prefer",
                        "discourage",
                        "conditional_require",
                        "conditional_forbid",
                    }
                ):
                    # This value is a deterministic normalization of an
                    # explicit constraint. Bind it to the exact constraint
                    # clause, never to an arbitrary proposal-level event.
                    refs = self._value_refs(candidate_content, events, source_uri)
                bindings[field_name] = refs
                all_refs.extend(refs)
        for field_name in (
            "semantic.speech_act",
            "semantic.commitment",
            "semantic.temporal_scope",
            "semantic.relation_to_existing",
            "transition",
        ):
            bindings[field_name] = base_refs
        return tuple(dict.fromkeys(all_refs)), bindings

    def _value_refs(self, value: Any, events: list[Any], source_uri: str) -> tuple[EvidenceRef, ...]:
        needles = self._evidence_needles(value)
        refs: list[EvidenceRef] = []
        for event in events:
            text = event.text()
            folded = text.casefold()
            for needle in needles:
                start = folded.find(needle.casefold())
                if start >= 0:
                    refs.append(
                        EvidenceRef.from_event(
                            event,
                            source_uri=source_uri,
                            span_start=start,
                            span_end=start + len(needle),
                        )
                    )
                    break
        return tuple(dict.fromkeys(refs))

    def _evidence_needles(self, value: Any) -> tuple[str, ...]:
        if isinstance(value, Mapping):
            return tuple(str(item) for item in value.values() if str(item))
        if isinstance(value, list | tuple | set):
            return tuple(str(item) for item in value if str(item))
        text = str(value).strip()
        return (text,) if text else ()

    def _semantic(self, candidate: MemoryCandidateDraft) -> tuple[str, str, str, str]:
        explicit = (
            candidate.fields.get("_semantic_speech_act"),
            candidate.fields.get("_semantic_commitment"),
            candidate.fields.get("_semantic_temporal_scope"),
            candidate.fields.get("_semantic_relation"),
        )
        if any(value is not None for value in explicit):
            speech, commitment, temporal, relation = (
                str(value or fallback)
                for value, fallback in zip(
                    explicit,
                    ("unknown", "unknown", "unknown", "ambiguous"),
                    strict=True,
                )
            )
            return speech, commitment, temporal, relation
        text = candidate.content.casefold()
        if candidate.memory_type == MemoryType.AGENT_EXPERIENCE:
            return "observation", "intended", "past", "supplements"
        if candidate.memory_type == MemoryType.PROJECT_DECISION and any(
            token in text for token in ("future", "later", "evaluate", "以后", "评估", "候选")
        ):
            speech = "evaluation_request" if any(token in text for token in ("evaluate", "评估")) else "proposal"
            return speech, "exploratory", "future", "alternative"
        if any(token in text for token in ("retract", "撤回", "不再")):
            return "retraction", "confirmed", "current", "corrects"
        return "confirmation", "confirmed", "current", "unrelated"

    def _epistemic(self, role: str) -> EpistemicStatus:
        if role == "user":
            return EpistemicStatus.EXPLICIT
        if role == "tool":
            return EpistemicStatus.OBSERVED
        return EpistemicStatus.INFERRED

    def _suggested_scopes(
        self, candidate: MemoryCandidateDraft, episode: EvidenceEpisode, user_id: str
    ) -> tuple[ScopeRef, ...]:
        principal = scope_from_external("user", user_id)
        primary = episode.origin.primary_scope
        if candidate.memory_type in {MemoryType.PROFILE, MemoryType.PREFERENCE}:
            return (principal,)
        if (
            candidate.memory_type
            in {
                MemoryType.PROJECT_RULE,
                MemoryType.PROJECT_DECISION,
                MemoryType.AGENT_EXPERIENCE,
            }
            and primary is not None
        ):
            return (primary,)
        return tuple(scope for scope in (primary, principal) if scope is not None)

    def _topic(self, text: str) -> str:
        stop = {
            "project",
            "rule",
            "memoryos",
            "must",
            "never",
            "keep",
            "run",
            "use",
            "before",
            "after",
            "remember",
            "prefer",
            "preference",
            "decided",
            "adopted",
            "architecture",
            "decision",
            "the",
            "this",
            "that",
            "with",
            "i",
            "we",
            "my",
            "our",
        }
        return next((token for token in self._tokens(text) if token.casefold() not in stop), "general")

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9_.-]+|[\u4e00-\u9fff]+", text)


class CanonicalMemoryFormationService:
    """串起证据校验、准入、身份解析、状态转换和事务规划。"""

    def __init__(
        self,
        source_store: SourceStore | None,
        *,
        relation_store: RelationStore | None = None,
        alias_registry: AliasRegistry | None = None,
    ) -> None:
        self.source_store = source_store
        self.relation_store = relation_store
        self.validator = ProposalEvidenceValidator()
        self.normalizer = MemorySemanticNormalizer()
        registry = MemoryTypeRegistry()
        eligibility = MemoryTypeEligibilityPolicy()
        self.admission = ProposalAdmissionGate(registry, eligibility_policy=eligibility)
        self.identity = StableMemoryIdentityResolver(alias_registry, registry)
        self.reconciler = MemorySemanticReconciler()
        self.transition = MemoryTransitionPolicy(registry, eligibility)
        self.transactions = MemoryTransactionPlanner()

    def stage(
        self,
        operations: tuple[ContextOperation, ...],
        staging: dict[str, ContextObject] | None = None,
    ) -> dict[str, ContextObject]:
        request_staging = staging if staging is not None else {}
        for operation in operations:
            payload = operation.payload.get("context_object")
            if isinstance(payload, dict):
                obj = ContextObject.from_dict(payload)
                request_staging[obj.uri] = obj
        return request_staging

    def plan(
        self,
        proposal: MemorySemanticProposal,
        *,
        archive: SessionArchive,
        episode: EvidenceEpisode,
        retrieval_views: list[str] | None = None,
        staged_objects: Mapping[str, ContextObject] | None = None,
        commit_group_id: str | None = None,
    ) -> CanonicalFormationResult:
        """Plan an ordinary proposal without a destructive-effect capability."""

        return self._plan(
            proposal,
            archive=archive,
            episode=episode,
            retrieval_views=retrieval_views,
            staged_objects=staged_objects,
            commit_group_id=commit_group_id,
            confirmed_pending=None,
        )

    def _plan(
        self,
        proposal: MemorySemanticProposal,
        *,
        archive: SessionArchive,
        episode: EvidenceEpisode,
        retrieval_views: list[str] | None = None,
        staged_objects: Mapping[str, ContextObject] | None = None,
        commit_group_id: str | None = None,
        confirmed_pending: PendingMemoryProposal | None,
    ) -> CanonicalFormationResult:
        """Internal planner used only by a validated pending-review resolution."""

        proposal = self._bind_evidence_context(self._bind_system_identity(proposal, episode))
        memory_scope = self._memory_scope(proposal, archive, episode)
        if proposal.metadata.get("fallback_pending_only") is True:
            normalized = self.normalizer.normalize(proposal)
            return self._pending_result(
                normalized,
                memory_scope=memory_scope,
                archive=archive,
                episode=episode,
                reason=CandidateProposalAdapter.FALLBACK_PENDING_REASON,
                retrieval_views=retrieval_views or [],
                commit_group_id=commit_group_id or "",
                staged_objects=staged_objects,
            )
        raw_validation = self.validator.validate(proposal, episode)
        grounded = raw_validation.proposal
        if raw_validation.valid and grounded.semantic_contract_version.casefold() == "v3":
            grounded = replace(
                grounded,
                metadata={
                    **dict(grounded.metadata),
                    "source_grounded_field_values": {
                        **{f"identity.{key}": value for key, value in grounded.identity_fields.items()},
                        **{f"value.{key}": value for key, value in grounded.value_fields.items()},
                    },
                },
            )
        normalized = self.normalizer.normalize(grounded)
        normalized = self._bind_system_resolved_replacement_target(
            normalized,
            memory_scope=memory_scope,
            archive=archive,
            episode=episode,
            staged_objects=staged_objects,
        )
        # Normalization can expose schema mismatches; validate the normalized
        # proposal again instead of reusing a pre-normalization boolean.
        normalized_validation = self.validator.validate(normalized, episode)
        validation = type(normalized_validation)(
            valid=raw_validation.valid and normalized_validation.valid,
            proposal=normalized_validation.proposal,
            errors=tuple(dict.fromkeys((*raw_validation.errors, *normalized_validation.errors))),
            unsupported_fields=tuple(
                dict.fromkeys((*raw_validation.unsupported_fields, *normalized_validation.unsupported_fields))
            ),
        )
        normalized = validation.proposal
        if normalized.metadata.get("identity_resolution_pending") is True:
            return self._pending_result(
                normalized,
                memory_scope=memory_scope,
                archive=archive,
                episode=episode,
                reason="PENDING_IDENTITY_RESOLUTION:missing_attribute_key",
                retrieval_views=retrieval_views or [],
                commit_group_id=commit_group_id or "",
                staged_objects=staged_objects,
            )
        admission = self.admission.evaluate(
            validation,
            episode=episode,
            memory_scope=memory_scope,
            source_role=str(proposal.metadata.get("source_role", "user")),
        )
        normalized = replace(
            normalized,
            metadata={
                **dict(normalized.metadata),
                "model_confidence": normalized.confidence,
                "admission_score": admission.admission_score,
                "admission_threshold": admission.admission_threshold,
                "admission_score_components": dict(admission.score_components),
            },
        )
        if admission.decision != ProposalAdmissionDecision.ACCEPT_FOR_RECONCILE:
            if admission.decision == ProposalAdmissionDecision.PENDING:
                return self._pending_result(
                    normalized,
                    memory_scope=memory_scope,
                    archive=archive,
                    episode=episode,
                    reason=admission.reason,
                    retrieval_views=retrieval_views or [],
                    commit_group_id=commit_group_id or "",
                    staged_objects=staged_objects,
                )
            return CanonicalFormationResult((), admission.decision, admission.reason, normalized)
        normalized = self._separate_display_fields(normalized)
        identity = self.identity.resolve(
            normalized,
            memory_scope,
            tenant_id=episode.tenant_id,
            owner_user_id=archive.user_id,
        )
        slot: MemorySlot | None = None
        claims: tuple[MemoryClaim, ...] = ()
        if self.source_store is not None or staged_objects:
            repository = CanonicalMemoryRepository._for_planning(
                self.source_store,
                self.relation_store,
                staged_objects or {},
            )
            slot, claims = repository.load(identity)
        target_state_error = self._related_active_target_error(normalized, slot, claims)
        if target_state_error:
            return self._pending_result(
                normalized,
                memory_scope=memory_scope,
                archive=archive,
                episode=episode,
                reason=f"semantic_reconciliation_pending:{target_state_error}",
                retrieval_views=retrieval_views or [],
                commit_group_id=commit_group_id or "",
                related_existing_memory_ids=normalized.all_related_memory_ids,
                staged_objects=staged_objects,
            )
        reconciled = self.reconciler.reconcile(normalized, identity, slot=slot, claims=claims)
        try:
            transition = (
                self.transition._apply_confirmed_pending_review(
                    confirmed_pending,
                    normalized,
                    identity,
                    reconciled,
                    authorization_id=stable_hash(
                        [confirmed_pending.uri, confirmed_pending.lifecycle_revision, normalized.fingerprint],
                        length=40,
                    ),
                    owner_user_id=archive.user_id,
                    tenant_id=episode.tenant_id,
                )
                if confirmed_pending is not None
                else self.transition.apply(normalized, identity, reconciled)
            )
        except PendingSemanticReconciliation as pending:
            if (
                pending.reason == "relation_requires_confirmation"
                and reconciled.active_claim is not None
                and isinstance(normalized.semantic, NormalizedSemanticAssessment)
            ):
                active = reconciled.active_claim
                normalized = replace(
                    normalized,
                    semantic=replace(
                        normalized.semantic,
                        relation_to_existing=SemanticRelation.SUPERSEDES,
                    ),
                    related_memory_ids=(active.uri,),
                    related_slot_ids=(identity.slot_id,),
                    related_claim_ids=(active.claim_id,),
                    metadata={
                        **dict(normalized.metadata),
                        "review_proposed_relation": SemanticRelation.SUPERSEDES.value,
                        "review_proposed_target_claim_uri": active.uri,
                    },
                )
            related_existing = tuple(
                dict.fromkeys(
                    (
                        *normalized.all_related_memory_ids,
                        *(
                            (reconciled.claim.claim_id, reconciled.claim.uri)
                            if reconciled.claim is not None
                            else ()
                        ),
                        *(
                            (reconciled.active_claim.claim_id, reconciled.active_claim.uri)
                            if reconciled.active_claim is not None
                            else ()
                        ),
                    )
                )
            )
            return self._pending_result(
                normalized,
                memory_scope=memory_scope,
                archive=archive,
                episode=episode,
                reason=f"semantic_reconciliation_pending:{pending.reason}",
                retrieval_views=retrieval_views or [],
                commit_group_id=commit_group_id or "",
                related_existing_memory_ids=related_existing,
                staged_objects=staged_objects,
            )
        plan = self.transactions.build(
            normalized,
            memory_scope,
            transition,
            tenant_id=episode.tenant_id,
            owner_user_id=archive.user_id,
            episode_id=episode.episode_id,
            commit_group_id=commit_group_id or "",
            planning_task_id=archive.task_id,
        )
        operations = plan.to_context_operations(
            user_id=archive.user_id,
            tenant_id=episode.tenant_id,
            episode_id=episode.episode_id,
        )
        self._decorate_operations(operations, normalized, retrieval_views or [])
        return CanonicalFormationResult(
            tuple(operations),
            admission.decision,
            admission.reason,
            normalized,
            resolved_identity=identity,
        )

    def _related_active_target_error(
        self,
        proposal: MemorySemanticProposal,
        slot: MemorySlot | None,
        claims: tuple[MemoryClaim, ...],
    ) -> str:
        """Bind high-impact relation identifiers to one exact repository ACTIVE Claim."""

        relation = str(
            getattr(proposal.semantic.relation_to_existing, "value", proposal.semantic.relation_to_existing)
        ).strip().casefold()
        if relation not in {"corrects", "supersedes", "supplements"}:
            return ""
        if slot is None or not slot.active_claim_id:
            return "relation_target_requires_active_claim"
        active = next((claim for claim in claims if claim.claim_id == slot.active_claim_id), None)
        if active is None or active.current.state != "ACTIVE":
            return "relation_target_requires_active_claim"
        if proposal.related_claim_ids and tuple(proposal.related_claim_ids) != (active.claim_id,):
            return "relation_claim_target_mismatch"
        if proposal.related_slot_ids and tuple(proposal.related_slot_ids) != (slot.slot_id,):
            return "relation_slot_target_mismatch"
        if proposal.related_memory_ids and tuple(proposal.related_memory_ids) not in {
            (active.claim_id,),
            (active.uri,),
        }:
            return "relation_memory_target_mismatch"
        if not proposal.related_claim_ids and not proposal.related_memory_ids:
            return "relation_target_missing"
        return ""

    def _bind_system_resolved_replacement_target(
        self,
        proposal: MemorySemanticProposal,
        *,
        memory_scope: MemoryScope,
        archive: SessionArchive,
        episode: EvidenceEpisode,
        staged_objects: Mapping[str, ContextObject] | None,
    ) -> MemorySemanticProposal:
        """Bind a rule-extracted replacement to the exact active claim in its resolved slot."""

        if proposal.metadata.get("replacement_explicit") is not True or proposal.all_related_memory_ids:
            return proposal
        relation = str(
            getattr(proposal.semantic.relation_to_existing, "value", proposal.semantic.relation_to_existing)
        ).casefold()
        if relation not in {"corrects", "supersedes"} or (self.source_store is None and not staged_objects):
            return proposal
        identity = self.identity.resolve(
            proposal,
            memory_scope,
            tenant_id=episode.tenant_id,
            owner_user_id=archive.user_id,
        )
        repository = CanonicalMemoryRepository._for_planning(
            self.source_store,
            self.relation_store,
            staged_objects or {},
        )
        slot, claims = repository.load(identity)
        if slot is None or slot.active_claim_id is None:
            return proposal
        active = next((claim for claim in claims if claim.claim_id == slot.active_claim_id), None)
        if active is None:
            return proposal
        return replace(
            proposal,
            related_memory_ids=(active.uri,),
            related_slot_ids=(slot.slot_id,),
            related_claim_ids=(active.claim_id,),
            metadata={
                **dict(proposal.metadata),
                "replacement_target_resolution": "system_exact_active_slot_claim",
            },
        )

    def _separate_display_fields(self, proposal: MemorySemanticProposal) -> MemorySemanticProposal:
        display_names = {
            "title",
            "display_name",
            "display_text",
            "source_text",
            "source_wording",
            "summary",
            "details",
            "rationale",
            "reason",
            "decision",
            "rule",
        }
        display = {
            key: value for key, value in proposal.value_fields.items() if key in display_names
        }
        if not display:
            return proposal
        remaining = {
            key: value for key, value in proposal.value_fields.items() if key not in display_names
        }
        display_evidence = {
            field_name: [ref.to_dict() for ref in refs]
            for field_name, refs in proposal.field_evidence_refs.items()
            if field_name.startswith("value.") and field_name.split(".", 1)[1] in display_names
        }
        remaining_evidence = {
            field_name: refs
            for field_name, refs in proposal.field_evidence_refs.items()
            if field_name not in display_evidence
        }
        return replace(
            proposal,
            value_fields=remaining,
            field_evidence_refs=remaining_evidence,
            metadata={
                **dict(proposal.metadata),
                "display_fields": display,
                "display_field_evidence_refs": display_evidence,
            },
        )

    def plan_pending(
        self,
        proposal: MemorySemanticProposal,
        *,
        archive: SessionArchive,
        episode: EvidenceEpisode,
        reason: str,
        retrieval_views: list[str] | None = None,
        commit_group_id: str | None = None,
        staged_objects: Mapping[str, ContextObject] | None = None,
    ) -> CanonicalFormationResult:
        """Persist a caller-admitted pending proposal without promoting it."""

        bound = self._bind_evidence_context(self._bind_system_identity(proposal, episode))
        normalized = self.normalizer.normalize(bound)
        return self._pending_result(
            normalized,
            memory_scope=self._memory_scope(normalized, archive, episode),
            archive=archive,
            episode=episode,
            reason=reason,
            retrieval_views=retrieval_views or [],
            commit_group_id=commit_group_id or "",
            staged_objects=staged_objects,
        )

    def plan_pending_lifecycle_transition(
        self,
        pending_uri: str,
        lifecycle_state: LifecycleState,
        *,
        tenant_id: str,
        owner_user_id: str,
        commit_group_id: str,
        reason: str = "",
        retry_increment: bool = False,
        updated_at: str = "",
        resolution_operations: tuple[ContextOperation, ...] = (),
        review_command_id: str = "",
        review_decision: str = "",
        review_request_digest: str = "",
    ) -> ContextOperation:
        """Plan a review lifecycle update; the caller must commit the returned operation."""

        if self.source_store is None:
            raise RuntimeError("pending lifecycle transitions require a SourceStore")
        current = CanonicalMemoryRepository(self.source_store, self.relation_store).load_pending(
            pending_uri,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
        )
        resolution_idempotency_keys: tuple[str, ...] = ()
        resolved_claim_uris: tuple[str, ...] = ()
        if lifecycle_state == LifecycleState.RESOLVED:
            resolution_idempotency_keys, resolved_claim_uris = self._pending_resolution_links(
                current,
                resolution_operations,
                tenant_id=tenant_id,
                owner_user_id=owner_user_id,
            )
        updated = current.with_lifecycle(
            lifecycle_state,
            updated_at=updated_at,
            retry_increment=retry_increment,
            reason=reason,
            review_command_id=review_command_id,
            review_decision=review_decision,
            review_request_digest=review_request_digest,
        )
        transition_from = current.lifecycle_state.value
        expected_lifecycle_revision = current.lifecycle_revision
        if updated is current and current.lifecycle_history:
            last_transition = dict(current.lifecycle_history[-1])
            if str(last_transition.get("to")) == lifecycle_state.value:
                transition_from = str(last_transition.get("from") or transition_from)
                expected_lifecycle_revision = int(
                    last_transition.get("from_revision", max(1, current.lifecycle_revision - 1))
                )
        obj = updated.to_context_object(tenant_id=tenant_id, owner_user_id=owner_user_id)
        idempotency_key = stable_hash(
            [
                commit_group_id,
                pending_uri,
                transition_from,
                updated.lifecycle_state.value,
                updated.retry_count,
                updated.lifecycle_revision,
                review_command_id,
                review_decision,
                review_request_digest,
            ],
            length=40,
        )
        operation = ContextOperation(
            context_type=obj.context_type,
            action=OperationAction.UPDATE,
            target_uri=obj.uri,
            user_id=owner_user_id,
            operation_id=f"op_{stable_hash(['pending_lifecycle', idempotency_key], length=32)}",
            evidence=[ref.to_dict() for ref in updated.proposal.evidence_refs],
            confidence=updated.proposal.confidence,
            source_uri=(updated.proposal.evidence_refs[0].source_uri if updated.proposal.evidence_refs else None),
            source_episode_id=(
                updated.proposal.evidence_refs[0].episode_id if updated.proposal.evidence_refs else None
            ),
            source_session_id=str(updated.proposal.metadata.get("source_session_id") or "") or None,
            created_at=updated.updated_at,
            payload={
                "canonical_pending_proposal": True,
                "pending_lifecycle_transition": True,
                "pending_proposal_id": updated.proposal_id,
                "pending_lifecycle_state": updated.lifecycle_state.value,
                "pending_lifecycle_reason": reason,
                "pending_review_binding": {
                    "command_id": review_command_id,
                    "decision": review_decision.strip().upper(),
                    "request_digest": review_request_digest,
                }
                if review_command_id or review_decision or review_request_digest
                else {},
                "expected_pending_lifecycle_state": transition_from,
                "expected_pending_lifecycle_revision": expected_lifecycle_revision,
                "expected_pending_updated_at": current.updated_at,
                "pending_lifecycle_revision": updated.lifecycle_revision,
                "pending_lifecycle_resolution": lifecycle_state == LifecycleState.RESOLVED,
                "resolution_idempotency_keys": list(resolution_idempotency_keys),
                "resolved_claim_uris": list(resolved_claim_uris),
                "idempotency_key": idempotency_key,
                "commit_group_id": commit_group_id,
                "tenant_id": tenant_id,
                "memory_type": updated.proposal.memory_type,
                "admission": {"decision": "pending", "reason": updated.pending_reason_code},
                "retrieval_views": list(updated.retrieval_views),
                "schema_version": PendingMemoryProposal.SCHEMA_VERSION,
                "context_object": obj.to_dict(),
                "content": updated.content(),
            },
        )
        if lifecycle_state == LifecycleState.RESOLVED:
            transaction_ids = {
                str(item.payload.get("transaction_id") or "")
                for item in resolution_operations
                if item.payload.get("canonical_memory") is True
            }
            idempotency_keys = {
                str(item.payload.get("idempotency_key") or "")
                for item in resolution_operations
                if item.payload.get("canonical_memory") is True
            }
            slot_ids = {
                str(item.payload.get("slot_id") or "")
                for item in resolution_operations
                if item.payload.get("canonical_memory") is True
            }
            if (
                len(transaction_ids) != 1
                or "" in transaction_ids
                or len(idempotency_keys) != 1
                or "" in idempotency_keys
                or len(slot_ids) != 1
                or "" in slot_ids
            ):
                raise ValueError("RESOLVED requires one canonical transaction, idempotency key, and slot")
            operation.payload.update(
                {
                    "canonical_memory": True,
                    "canonical_pending_resolution": True,
                    "transaction_id": next(iter(transaction_ids)),
                    "idempotency_key": next(iter(idempotency_keys)),
                    "slot_id": next(iter(slot_ids)),
                    "identity_algorithm_version": IDENTITY_ALGORITHM_V2,
                }
            )
        return operation

    def plan_confirmed_pending_resolution(
        self,
        pending_uri: str,
        confirmed_proposal: MemorySemanticProposal,
        *,
        archive: SessionArchive,
        episode: EvidenceEpisode,
        tenant_id: str,
        owner_user_id: str,
        commit_group_id: str,
        retrieval_views: list[str] | None = None,
        reason: str = "canonical_confirmation_committed",
        review_command_id: str = "",
        review_decision: str = "",
        review_request_digest: str = "",
    ) -> CanonicalFormationResult:
        """Plan one canonical confirmation followed by a linked pending resolution."""

        if self.source_store is None:
            raise RuntimeError("pending resolution requires a SourceStore")
        if archive.user_id != owner_user_id or episode.tenant_id != tenant_id:
            raise PermissionError("pending resolution archive boundary does not match owner or tenant")
        committed_pending = read_committed_pending(
            self.source_store,
            pending_uri,
            self.relation_store,
        )
        pending = PendingMemoryProposal.from_context_object(committed_pending.object)
        if (
            str(committed_pending.object.tenant_id or "default") != tenant_id
            or committed_pending.object.owner_user_id != owner_user_id
        ):
            raise FileNotFoundError(pending_uri)
        if pending.lifecycle_state != LifecycleState.CONFIRMED:
            raise ValueError("pending proposal must be CONFIRMED before canonical resolution")
        head = dict(committed_pending.head or {})
        receipt = dict(committed_pending.receipt or {})
        confirm_operations = [
            item
            for item in receipt.get("operations", []) or []
            if isinstance(item, dict)
            and str(item.get("operation_id") or "") == str(head.get("current_operation_id") or "")
            and str(item.get("target_uri") or "") == pending_uri
        ]
        if len(confirm_operations) != 1:
            raise ValueError("CONFIRMED pending has no unique committed confirmation operation")
        confirm_payload = dict(confirm_operations[0].get("payload", {}) or {})
        if (
            confirm_payload.get("pending_lifecycle_transition") is not True
            or str(confirm_payload.get("pending_lifecycle_state") or "").upper() != "CONFIRMED"
            or str(head.get("proposal_fingerprint") or "") != pending.proposal.fingerprint
            or not str(head.get("receipt_digest") or "")
        ):
            raise ValueError("CONFIRMED pending lifecycle proof is not an authorization receipt")
        if pending.proposal.memory_type != confirmed_proposal.memory_type:
            raise ValueError("confirmed proposal memory type does not match pending proposal")
        comparable_pending = self._separate_display_fields(self.normalizer.normalize(pending.proposal))
        comparable_confirmed = self._separate_display_fields(self.normalizer.normalize(confirmed_proposal))
        if (
            dict(comparable_pending.identity_fields) != dict(comparable_confirmed.identity_fields)
            or dict(comparable_pending.value_fields) != dict(comparable_confirmed.value_fields)
            or dict(comparable_pending.metadata.get("display_fields", {}) or {})
            != dict(comparable_confirmed.metadata.get("display_fields", {}) or {})
        ):
            raise ValueError("confirmed proposal cannot rewrite pending identity or value fields")
        pending_relation = str(
            getattr(
                comparable_pending.semantic.relation_to_existing,
                "value",
                comparable_pending.semantic.relation_to_existing,
            )
        ).casefold()
        confirmed_relation = str(
            getattr(
                comparable_confirmed.semantic.relation_to_existing,
                "value",
                comparable_confirmed.semantic.relation_to_existing,
            )
        ).casefold()
        if pending_relation != confirmed_relation:
            raise ValueError("confirmed proposal cannot change pending semantic relation")
        pending_targets = (
            tuple(comparable_pending.related_memory_ids),
            tuple(comparable_pending.related_slot_ids),
            tuple(comparable_pending.related_claim_ids),
        )
        confirmed_targets = (
            tuple(comparable_confirmed.related_memory_ids),
            tuple(comparable_confirmed.related_slot_ids),
            tuple(comparable_confirmed.related_claim_ids),
        )
        if pending_targets != confirmed_targets:
            raise ValueError("confirmed proposal cannot change pending relation targets")
        stable_semantic_fields = (
            "temporal_scope",
            "utterance_mode",
            "attribution",
            "durability",
            "modal_force",
            "atomicity",
        )
        if any(
            str(
                getattr(
                    getattr(comparable_pending.semantic, field_name),
                    "value",
                    getattr(comparable_pending.semantic, field_name),
                )
            )
            != str(
                getattr(
                    getattr(comparable_confirmed.semantic, field_name),
                    "value",
                    getattr(comparable_confirmed.semantic, field_name),
                )
            )
            for field_name in stable_semantic_fields
        ):
            raise ValueError("confirmed proposal cannot change pending proposition semantics")
        formed = self._plan(
            confirmed_proposal,
            archive=archive,
            episode=episode,
            retrieval_views=retrieval_views or list(pending.retrieval_views),
            commit_group_id=commit_group_id,
            confirmed_pending=replace(pending, proposal=comparable_pending),
        )
        canonical_operations = tuple(
            operation for operation in formed.operations if operation.payload.get("canonical_memory") is True
        )
        if formed.decision != ProposalAdmissionDecision.ACCEPT_FOR_RECONCILE or not canonical_operations:
            raise ValueError(
                "confirmed pending resolution requires a canonical state-changing transaction: "
                f"{formed.decision.value}:{formed.reason}"
            )
        pending_identity = self.identity.resolve(
            pending.proposal,
            pending.scope,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
        )
        confirmed_scope = self._memory_scope(formed.proposal, archive, episode)
        confirmed_identity = self.identity.resolve(
            formed.proposal,
            confirmed_scope,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
        )
        if (
            pending_identity.slot_id != confirmed_identity.slot_id
            or pending_identity.claim_id != confirmed_identity.claim_id
        ):
            raise ValueError("confirmed proposal identity does not match pending proposal")
        resolution = self.plan_pending_lifecycle_transition(
            pending_uri,
            LifecycleState.RESOLVED,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            commit_group_id=commit_group_id,
            reason=reason,
            resolution_operations=canonical_operations,
            review_command_id=review_command_id,
            review_decision=review_decision,
            review_request_digest=review_request_digest,
        )
        resolution.payload["confirmation_receipt_digest"] = str(head["receipt_digest"])
        resolution.payload["confirmation_operation_id"] = str(head["current_operation_id"])
        resolution.payload["confirmation_lifecycle_revision"] = pending.lifecycle_revision
        # Confirmation/application is a new deterministic authorization plan,
        # not a replay of the LLM envelope that originally created pending.
        # The committer publishes one immutable direct planning proof whose
        # operation digest includes the review command and CONFIRM receipt.
        for operation in (*canonical_operations, resolution):
            operation.payload.pop("planning_task_id", None)
            operation.payload.pop("planning_digest", None)
        return CanonicalFormationResult(
            (*canonical_operations, resolution),
            formed.decision,
            formed.reason,
            formed.proposal,
        )

    def plan_pending_correction(
        self,
        pending_uri: str,
        corrected_proposal: MemorySemanticProposal,
        *,
        archive: SessionArchive,
        episode: EvidenceEpisode,
        tenant_id: str,
        owner_user_id: str,
        commit_group_id: str,
        retrieval_views: list[str] | None = None,
        reason: str = "corrected_by_new_proposal",
        review_command_id: str = "",
        review_decision: str = "",
        review_request_digest: str = "",
    ) -> CanonicalFormationResult:
        """Atomically commit a corrected proposal and terminalize its predecessor.

        This is deliberately separate from review confirmation.  Reasons such
        as missing evidence/schema/scope cannot be authorized by changing a
        lifecycle flag; they require a different, independently validated
        proposal in a new commit group.
        """

        if self.source_store is None:
            raise RuntimeError("pending correction requires a SourceStore")
        if archive.user_id != owner_user_id or episode.tenant_id != tenant_id:
            raise PermissionError("pending correction archive boundary does not match owner or tenant")
        committed = read_committed_pending(self.source_store, pending_uri, self.relation_store)
        pending = PendingMemoryProposal.from_context_object(committed.object)
        if (
            str(committed.object.tenant_id or "default") != tenant_id
            or committed.object.owner_user_id != owner_user_id
        ):
            raise FileNotFoundError(pending_uri)
        if pending.lifecycle_state not in {LifecycleState.PENDING, LifecycleState.RETRYABLE}:
            raise ValueError("only a live pending proposal can be replaced by a corrected proposal")
        policy = pending.reason_policy
        if not policy.requires_new_proposal:
            raise ValueError("reviewable pending reasons must use structured review, not correction")
        if not commit_group_id or commit_group_id == pending.request_identity:
            raise ValueError("pending correction requires a new commit group")
        if policy.requires_reextraction and archive.task_id == pending.request_identity:
            raise ValueError("fallback correction requires re-extraction in a new task")
        if corrected_proposal.fingerprint == pending.proposal.fingerprint:
            raise ValueError("pending correction requires a new proposal fingerprint")

        linked = replace(
            corrected_proposal,
            metadata={
                **dict(corrected_proposal.metadata),
                "corrects_pending_uri": pending_uri,
                "corrects_pending_fingerprint": pending.proposal.fingerprint,
                "correction_commit_group_id": commit_group_id,
                "correction_reextracted": policy.requires_reextraction,
            },
        )
        formed = self._plan(
            linked,
            archive=archive,
            episode=episode,
            retrieval_views=retrieval_views or list(pending.retrieval_views),
            commit_group_id=commit_group_id,
            confirmed_pending=None,
        )
        canonical_operations = tuple(
            operation
            for operation in formed.operations
            if operation.payload.get("canonical_memory") is True
        )
        if formed.decision != ProposalAdmissionDecision.ACCEPT_FOR_RECONCILE or not canonical_operations:
            raise ValueError("corrected proposal must independently pass validation and change canonical state")

        transaction_ids = {str(item.payload.get("transaction_id") or "") for item in canonical_operations}
        idempotency_keys = {str(item.payload.get("idempotency_key") or "") for item in canonical_operations}
        slot_ids = {str(item.payload.get("slot_id") or "") for item in canonical_operations}
        if (
            len(transaction_ids) != 1
            or "" in transaction_ids
            or len(idempotency_keys) != 1
            or "" in idempotency_keys
            or len(slot_ids) != 1
            or "" in slot_ids
        ):
            raise ValueError("corrected proposal must form exactly one canonical transaction")
        corrected_claim_uris = tuple(
            str(payload.get("uri") or "")
            for operation in canonical_operations
            if isinstance((payload := operation.payload.get("context_object")), dict)
            and dict(payload.get("metadata", {}) or {}).get("canonical_kind") == "claim"
            and dict(payload.get("metadata", {}) or {}).get("state") == "ACTIVE"
        )
        if not corrected_claim_uris:
            raise ValueError("corrected proposal must produce a linked ACTIVE Claim")

        terminal = self.plan_pending_lifecycle_transition(
            pending_uri,
            LifecycleState.REJECTED,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            commit_group_id=commit_group_id,
            reason=f"{reason}:{formed.proposal.fingerprint}",
            review_command_id=review_command_id,
            review_decision=review_decision,
            review_request_digest=review_request_digest,
        )
        transaction_id = next(iter(transaction_ids))
        idempotency_key = next(iter(idempotency_keys))
        slot_id = next(iter(slot_ids))
        terminal.payload.update(
            {
                "canonical_memory": True,
                "canonical_pending_correction": True,
                "transaction_id": transaction_id,
                "idempotency_key": idempotency_key,
                "slot_id": slot_id,
                "identity_algorithm_version": IDENTITY_ALGORITHM_V2,
                "corrected_proposal_id": formed.proposal.proposal_id,
                "corrected_proposal_fingerprint": formed.proposal.fingerprint,
                "corrected_claim_uris": list(corrected_claim_uris),
                "predecessor_proposal_fingerprint": pending.proposal.fingerprint,
                "correction_requires_reextraction": policy.requires_reextraction,
                "correction_task_id": archive.task_id,
            }
        )
        for operation in canonical_operations:
            operation.payload.update(
                {
                    "corrects_pending_uri": pending_uri,
                    "corrects_pending_fingerprint": pending.proposal.fingerprint,
                    "correction_commit_group_id": commit_group_id,
                }
            )
            payload = operation.payload.get("context_object")
            if isinstance(payload, dict):
                metadata = dict(payload.get("metadata", {}) or {})
                metadata.update(
                    {
                        "corrects_pending_uri": pending_uri,
                        "corrects_pending_fingerprint": pending.proposal.fingerprint,
                    }
                )
                payload["metadata"] = metadata
        return CanonicalFormationResult(
            (*canonical_operations, terminal),
            formed.decision,
            formed.reason,
            formed.proposal,
        )

    def _pending_resolution_links(
        self,
        pending: PendingMemoryProposal,
        operations: tuple[ContextOperation, ...],
        *,
        tenant_id: str,
        owner_user_id: str,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        keys: list[str] = []
        active_claim_uris: list[str] = []
        for operation in operations:
            if operation.payload.get("canonical_memory") is not True:
                continue
            key = str(operation.payload.get("idempotency_key") or "")
            payload = operation.payload.get("context_object")
            if not key or not isinstance(payload, dict):
                continue
            metadata = dict(payload.get("metadata", {}) or {})
            if (
                metadata.get("canonical_kind") != "claim"
                or metadata.get("state") != "ACTIVE"
                or str(payload.get("owner_user_id") or "") != owner_user_id
                or str(payload.get("tenant_id") or "default") != tenant_id
                or str(metadata.get("memory_type") or "") != pending.proposal.memory_type
            ):
                continue
            keys.append(key)
            active_claim_uris.append(str(payload.get("uri") or ""))
        resolved_claim_uris = tuple(dict.fromkeys(uri for uri in active_claim_uris if uri))
        if not resolved_claim_uris:
            raise ValueError("RESOLVED requires a linked canonical ACTIVE Claim operation")
        return tuple(dict.fromkeys(keys)), resolved_claim_uris

    def _pending_result(
        self,
        proposal: MemorySemanticProposal,
        *,
        memory_scope: MemoryScope,
        archive: SessionArchive,
        episode: EvidenceEpisode,
        reason: str,
        retrieval_views: list[str],
        commit_group_id: str,
        related_existing_memory_ids: tuple[str, ...] = (),
        staged_objects: Mapping[str, ContextObject] | None = None,
    ) -> CanonicalFormationResult:
        commit_group_id = commit_group_id or f"commit_group_{archive.task_id}"
        record = PendingMemoryProposal.create(
            proposal,
            memory_scope,
            tenant_id=episode.tenant_id,
            owner_user_id=archive.user_id,
            source_role=str(proposal.metadata.get("source_role", "user")),
            pending_reason_code=reason,
            request_identity=str(archive.task_id or episode.episode_id),
            related_existing_memory_ids=related_existing_memory_ids,
            retrieval_views=tuple(retrieval_views),
            created_at=archive.created_at,
        )
        staged_obj = (staged_objects or {}).get(record.uri)
        if staged_obj is not None:
            staged_record = PendingMemoryProposal.from_context_object(
                ContextObject.from_dict(staged_obj.to_dict())
            )
            if (
                str(staged_obj.tenant_id or "default") != episode.tenant_id
                or str(staged_obj.owner_user_id or "") != archive.user_id
                or staged_record.proposal.fingerprint != record.proposal.fingerprint
            ):
                raise ValueError("request-local pending staging crosses its proposal boundary")
            return CanonicalFormationResult(
                (),
                ProposalAdmissionDecision.PENDING,
                reason,
                proposal,
                pending_uri=staged_record.uri,
                pending_lifecycle_state=staged_record.lifecycle_state.value,
                pending_lifecycle_revision=staged_record.lifecycle_revision,
                pending_existing=True,
            )
        if self.source_store is not None:
            try:
                existing_record = CanonicalMemoryRepository(
                    self.source_store,
                    self.relation_store,
                ).load_pending(
                    record.uri,
                    tenant_id=episode.tenant_id,
                    owner_user_id=archive.user_id,
                )
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                existing_record = None
            if existing_record is not None:
                record = existing_record
                return CanonicalFormationResult(
                    (),
                    ProposalAdmissionDecision.PENDING,
                    reason,
                    proposal,
                    pending_uri=record.uri,
                    pending_lifecycle_state=record.lifecycle_state.value,
                    pending_lifecycle_revision=record.lifecycle_revision,
                    pending_existing=True,
                )
        obj = record.to_context_object(tenant_id=episode.tenant_id, owner_user_id=archive.user_id)
        idempotency_key = stable_hash(
            [commit_group_id, episode.tenant_id, archive.user_id, record.uri],
            length=40,
        )
        operation = ContextOperation(
            context_type=obj.context_type,
            action=OperationAction.ADD,
            target_uri=obj.uri,
            user_id=archive.user_id,
            operation_id=f"op_{stable_hash(['canonical_pending', idempotency_key], length=32)}",
            source_episode_id=episode.episode_id,
            source_session_id=archive.session_id,
            evidence=[ref.to_dict() for ref in proposal.evidence_refs],
            confidence=proposal.confidence,
            created_at=archive.created_at,
            payload={
                "canonical_pending_proposal": True,
                "pending_proposal_id": record.proposal_id,
                "idempotency_key": idempotency_key,
                "commit_group_id": commit_group_id,
                "planning_task_id": archive.task_id,
                "tenant_id": episode.tenant_id,
                "memory_type": record.proposal.memory_type,
                "admission": {"decision": "pending", "reason": record.pending_reason_code},
                "retrieval_views": list(record.retrieval_views),
                "source_roles": [record.source_role],
                "schema_version": PendingMemoryProposal.SCHEMA_VERSION,
                "context_object": obj.to_dict(),
                "content": record.content(),
            },
        )
        return CanonicalFormationResult(
            (operation,),
            ProposalAdmissionDecision.PENDING,
            reason,
            proposal,
            pending_uri=record.uri,
            pending_lifecycle_state=record.lifecycle_state.value,
            pending_lifecycle_revision=record.lifecycle_revision,
        )

    def _bind_system_identity(
        self,
        proposal: MemorySemanticProposal,
        episode: EvidenceEpisode,
    ) -> MemorySemanticProposal:
        if proposal.memory_type != MemoryType.EVENT.value:
            return proposal
        event_ids = sorted({ref.event_id for ref in proposal.evidence_refs if episode.event(ref.event_id) is not None})
        if not event_ids:
            return proposal
        metadata = dict(proposal.metadata)
        system_fields = {str(item) for item in metadata.get("system_identity_fields", []) or []}
        system_fields.add("event_key")
        metadata["system_identity_fields"] = sorted(system_fields)
        return replace(
            proposal,
            identity_fields={
                **dict(proposal.identity_fields),
                "event_key": f"{episode.episode_id}:{','.join(event_ids)}",
            },
            field_evidence_refs={
                **dict(proposal.field_evidence_refs),
                "identity.event_key": proposal.evidence_refs,
            },
            metadata=metadata,
        )

    def _bind_evidence_context(self, proposal: MemorySemanticProposal) -> MemorySemanticProposal:
        metadata = dict(proposal.metadata)
        transition_refs = tuple(proposal.field_evidence_refs.get("transition", ()))
        occurred = sorted(str(ref.occurred_at) for ref in transition_refs if ref.occurred_at)
        ingested = sorted(str(ref.ingested_at) for ref in transition_refs if ref.ingested_at)
        actor_ids = sorted({str(ref.actor_id) for ref in transition_refs if ref.actor_id})
        if occurred:
            metadata["effective_at"] = occurred[-1]
        if ingested:
            metadata["evidence_ingested_at"] = ingested[-1]
        if len(actor_ids) == 1:
            metadata.setdefault("asserted_by", actor_ids[0])
        return replace(proposal, metadata=metadata)

    def _memory_scope(
        self,
        proposal: MemorySemanticProposal,
        archive: SessionArchive,
        episode: EvidenceEpisode,
    ) -> MemoryScope:
        legal = {scope.key: scope for scope in episode.legal_scope_candidates()}
        # Use the archive-derived object, not model-supplied confidence/source
        # metadata for an otherwise matching key.
        suggested = [legal[scope.key] for scope in proposal.suggested_scope_refs if scope.key in legal]
        principal = scope_from_external(
            "user",
            archive.user_id,
            source=ScopeResolutionSource.EVENT,
        )
        if proposal.memory_type in {"profile", "preference"}:
            selected = [principal, *[scope for scope in suggested if scope.kind in {"workspace", "environment"}]]
            canonical_subject = principal
        elif suggested:
            selected = suggested
            canonical_subject = self._canonical_subject(proposal.memory_type, suggested)
        elif episode.origin.primary_scope is not None:
            selected = [episode.origin.primary_scope]
            canonical_subject = episode.origin.primary_scope
        else:
            selected = [principal]
            canonical_subject = principal
        principal_subject = canonical_subject.kind == "principal"
        return MemoryScope(
            applicability=ScopeSelector(tuple({scope.key: scope for scope in selected}.values())),
            visibility=VisibilityPolicy(
                episode.tenant_id,
                allowed_principal_ids=(archive.user_id,) if principal_subject else (),
                private=principal_subject,
            ),
            origin_refs=episode.origin.scope_refs,
            canonical_subject=canonical_subject,
            authority=AuthorityPolicy(
                principal_ids=(archive.user_id,),
                inferred=canonical_subject.inferred,
            ),
        )

    def _canonical_subject(self, memory_type: str, candidates: list[ScopeRef]) -> ScopeRef:
        priorities = (
            ("workspace", "environment", "asset", "location", "principal", "global")
            if memory_type in {"project_rule", "project_decision", "agent_experience"}
            else ("asset", "location", "workspace", "environment", "principal", "global")
        )
        subject = next((scope for kind in priorities for scope in candidates if scope.kind == kind), None)
        if subject is None:
            raise ValueError("canonical subject cannot be resolved from legal scope candidates")
        return subject

    def _decorate_operations(
        self,
        operations: list[ContextOperation],
        proposal: MemorySemanticProposal,
        retrieval_views: list[str],
    ) -> None:
        for operation in operations:
            object_payload = operation.payload.get("context_object")
            if not isinstance(object_payload, dict):
                continue
            metadata = dict(object_payload.get("metadata", {}) or {})
            metadata["retrieval_views"] = list(retrieval_views)
            metadata["model_confidence"] = proposal.confidence
            metadata["admission_score"] = float(proposal.metadata.get("admission_score", 0.0) or 0.0)
            metadata["admission_threshold"] = float(
                proposal.metadata.get("admission_threshold", 0.0) or 0.0
            )
            metadata["admission_score_components"] = canonicalize(
                proposal.metadata.get("admission_score_components", {}) or {}
            )
            metadata["admission"] = {
                "decision": "accept",
                "confidence": proposal.confidence,
                "model_confidence": proposal.confidence,
                "system_score": metadata["admission_score"],
                "threshold": metadata["admission_threshold"],
            }
            metadata["extractor_version"] = proposal.extractor_version
            metadata["model_id"] = proposal.model_id
            metadata["prompt_version"] = proposal.prompt_version
            materialized_revision = (
                materialized_current_revision_payload(metadata)
                if metadata.get("canonical_kind") == "claim"
                else {}
            )
            materialized_qualifiers = dict(materialized_revision.get("qualifiers", {}) or {})
            materialized_display = dict(materialized_qualifiers.get("display_fields", {}) or {})
            materialized_display_evidence = dict(
                materialized_qualifiers.get("display_field_evidence_refs", {}) or {}
            )
            # Top-level display metadata is a derived compatibility mirror.
            # Projection/retrieval use the materialized revision as truth.
            metadata["display_fields"] = canonicalize(materialized_display)
            metadata["display_field_evidence_refs"] = canonicalize(materialized_display_evidence)
            metadata["proposal_identity_fields"] = dict(proposal.identity_fields)
            metadata["merge_key"] = str(metadata.get("claim_id") or metadata.get("slot_id") or "")
            metadata["source_adapter_id"] = str(proposal.metadata.get("source_adapter_id", ""))
            metadata["source_session_id"] = str(proposal.metadata.get("source_session_id", ""))
            metadata["source_roles"] = [str(proposal.metadata.get("source_role", "user"))]
            metadata["source"] = {
                "adapter_id": metadata["source_adapter_id"],
                "session_id": metadata["source_session_id"],
                "roles": metadata["source_roles"],
            }
            metadata["connect"] = canonicalize(proposal.metadata.get("source_connect", {}) or {})
            object_payload["metadata"] = metadata
            operation.payload.update(
                {
                    "memory_type": proposal.memory_type,
                    "admission": metadata["admission"],
                    "retrieval_views": list(retrieval_views),
                    "schema_version": "canonical_memory_v2",
                    "source_adapter_id": metadata["source_adapter_id"],
                    "source_session_id": metadata["source_session_id"],
                    "source_roles": metadata["source_roles"],
                    "merge_key": metadata["merge_key"],
                    "merge_decision": "ADD" if operation.payload.get("expected_revision", 0) == 0 else "UPDATE",
                    "existing_uri": operation.target_uri if operation.payload.get("expected_revision", 0) != 0 else "",
                    "merge_reason": "canonical_transition_policy",
                }
            )
