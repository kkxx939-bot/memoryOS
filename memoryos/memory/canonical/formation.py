from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, cast

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.core.ids import stable_hash
from memoryos.memory.canonical.admission import (
    ProposalAdmissionDecision,
    ProposalAdmissionGate,
)
from memoryos.memory.canonical.episode import EvidenceEpisode
from memoryos.memory.canonical.evidence import EvidenceRef, ProposalEvidenceValidator
from memoryos.memory.canonical.identity import AliasRegistry, StableMemoryIdentityResolver
from memoryos.memory.canonical.proposal import (
    EpistemicStatus,
    MemorySemanticProposal,
    SemanticAssessment,
)
from memoryos.memory.canonical.reconcile import MemorySemanticReconciler
from memoryos.memory.canonical.repository import CanonicalMemoryRepository
from memoryos.memory.canonical.scope import (
    MemoryScope,
    ScopeRef,
    ScopeSelector,
    VisibilityPolicy,
    scope_from_external,
)
from memoryos.memory.canonical.semantic import MemorySemanticNormalizer
from memoryos.memory.canonical.state import MemoryClaim
from memoryos.memory.canonical.transaction import MemoryTransactionPlanner
from memoryos.memory.canonical.transition import MemoryTransitionPolicy
from memoryos.memory.schema import MemoryCandidateDraft, MemoryType
from memoryos.operations.model.context_operation import ContextOperation


@dataclass(frozen=True)
class CanonicalFormationResult:
    operations: tuple[ContextOperation, ...]
    decision: ProposalAdmissionDecision
    reason: str
    proposal: MemorySemanticProposal


class LegacyCandidateProposalAdapter:
    def adapt(
        self,
        candidate: MemoryCandidateDraft,
        episode: EvidenceEpisode,
        archive: SessionArchive,
    ) -> MemorySemanticProposal:
        events = [episode.event(event_id) for event_id in candidate.source_message_ids]
        matched = [event for event in events if event is not None]
        if not matched and episode.events:
            matched = [episode.events[0]]
        evidence_refs = tuple(EvidenceRef.from_event(event, source_uri=archive.archive_uri) for event in matched)
        identity, system_fields = self._identity(candidate, episode)
        speech, commitment, temporal, relation = self._semantic(candidate)
        epistemic = self._epistemic(candidate.source_role)
        return MemorySemanticProposal(
            proposal_id=f"proposal_{stable_hash([archive.task_id, candidate.memory_type.value, identity, candidate.fields, candidate.source_message_ids], length=32)}",
            memory_type=candidate.memory_type.value,
            identity_fields=identity,
            value_fields=self._values(candidate),
            semantic=SemanticAssessment(speech, commitment, temporal, relation),
            epistemic_status=epistemic,
            suggested_scope_refs=self._suggested_scopes(candidate, episode, archive.user_id),
            related_memory_ids=(),
            evidence_refs=evidence_refs,
            confidence=candidate.confidence,
            extractor_version="legacy_candidate_adapter_v1",
            model_id=None,
            metadata={
                "source_role": candidate.source_role,
                "source_adapter_id": candidate.source_adapter_id,
                "source_session_id": candidate.source_session_id or archive.session_id,
                "system_identity_fields": system_fields,
                "legacy_reason": candidate.reason,
            },
        )

    def _identity(self, candidate: MemoryCandidateDraft, episode: EvidenceEpisode) -> tuple[dict[str, Any], list[str]]:
        fields = candidate.fields
        topic = self._topic(candidate.content)
        if candidate.memory_type == MemoryType.PROFILE:
            return {"attribute_key": str(fields.get("attribute_key") or topic)}, []
        if candidate.memory_type == MemoryType.PREFERENCE:
            subject = str(fields.get("subject") or topic)
            dimension = str(fields.get("dimension") or topic)
            return {"subject": subject, "dimension": dimension}, []
        if candidate.memory_type == MemoryType.ENTITY:
            return {
                "entity_type": str(fields.get("entity_type") or fields.get("type") or "entity"),
                "canonical_entity_id": str(fields.get("canonical_entity_id") or fields.get("name") or topic),
            }, []
        if candidate.memory_type == MemoryType.PROJECT_RULE:
            return {"rule_topic": str(fields.get("rule_topic") or fields.get("rule_key") or topic)}, []
        if candidate.memory_type == MemoryType.PROJECT_DECISION:
            return {"decision_topic": str(fields.get("decision_topic") or fields.get("decision_key") or topic)}, []
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
        }, ["environment_signature"]

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
            if field_name not in {"project_id", "adapter_id", "tenant_id", "scope"}
        }
        return {key: candidate.content, **semantic_fields}

    def _semantic(self, candidate: MemoryCandidateDraft) -> tuple[str, str, str, str]:
        text = candidate.content.casefold()
        if candidate.memory_type == MemoryType.AGENT_EXPERIENCE:
            return "observation", "intended", "past", "supplements"
        if candidate.memory_type == MemoryType.PROJECT_DECISION and any(
            token in text for token in ("future", "later", "evaluate", "以后", "评估", "候选")
        ):
            return "proposal", "exploratory", "future", "alternative"
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
    def __init__(self, source_store: SourceStore | None, *, alias_registry: AliasRegistry | None = None) -> None:
        self.source_store = source_store
        self.validator = ProposalEvidenceValidator()
        self.normalizer = MemorySemanticNormalizer()
        self.admission = ProposalAdmissionGate()
        self.identity = StableMemoryIdentityResolver(alias_registry)
        self.reconciler = MemorySemanticReconciler()
        self.transition = MemoryTransitionPolicy()
        self.transactions = MemoryTransactionPlanner()
        self._planning_objects: dict[str, ContextObject] = {}

    def begin_planning(self) -> None:
        self._planning_objects.clear()

    def stage(self, operations: tuple[ContextOperation, ...]) -> None:
        for operation in operations:
            payload = operation.payload.get("context_object")
            if isinstance(payload, dict):
                obj = ContextObject.from_dict(payload)
                self._planning_objects[obj.uri] = obj

    def plan(
        self,
        proposal: MemorySemanticProposal,
        *,
        archive: SessionArchive,
        episode: EvidenceEpisode,
        retrieval_views: list[str] | None = None,
    ) -> CanonicalFormationResult:
        proposal = self._bind_system_identity(proposal, episode)
        memory_scope = self._memory_scope(proposal, archive, episode)
        validation = self.validator.validate(proposal, episode)
        normalized = self.normalizer.normalize(validation.proposal)
        validation = type(validation)(
            valid=validation.valid,
            proposal=normalized,
            errors=validation.errors,
            unsupported_fields=validation.unsupported_fields,
        )
        admission = self.admission.evaluate(
            validation,
            episode=episode,
            memory_scope=memory_scope,
            source_role=str(proposal.metadata.get("source_role", "user")),
        )
        if admission.decision != ProposalAdmissionDecision.ACCEPT_FOR_RECONCILE:
            return CanonicalFormationResult((), admission.decision, admission.reason, normalized)
        identity = self.identity.resolve(
            normalized,
            memory_scope,
            tenant_id=episode.tenant_id,
            owner_user_id=archive.user_id,
        )
        slot = None
        claims: tuple[MemoryClaim, ...] = ()
        if self.source_store is not None or self._planning_objects:
            slot, claims = CanonicalMemoryRepository(cast(SourceStore, self._planning_source())).load(identity)
        reconciled = self.reconciler.reconcile(normalized, identity, slot=slot, claims=claims)
        transition = self.transition.apply(normalized, identity, reconciled)
        plan = self.transactions.build(
            normalized,
            memory_scope,
            transition,
            tenant_id=episode.tenant_id,
            owner_user_id=archive.user_id,
            episode_id=episode.episode_id,
        )
        operations = plan.to_context_operations(
            user_id=archive.user_id,
            tenant_id=episode.tenant_id,
            episode_id=episode.episode_id,
        )
        self._compatibility_metadata(operations, normalized, retrieval_views or [])
        return CanonicalFormationResult(tuple(operations), admission.decision, admission.reason, normalized)

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
            identity_fields={**dict(proposal.identity_fields), "event_key": f"{episode.episode_id}:{','.join(event_ids)}"},
            metadata=metadata,
        )

    def _planning_source(self):  # noqa: ANN202
        service = self

        class PlanningSource:
            def read_object(self, uri: str) -> ContextObject:
                if uri in service._planning_objects:
                    return service._planning_objects[uri]
                if service.source_store is None:
                    raise FileNotFoundError(uri)
                return service.source_store.read_object(uri)

        return PlanningSource()

    def _memory_scope(
        self,
        proposal: MemorySemanticProposal,
        archive: SessionArchive,
        episode: EvidenceEpisode,
    ) -> MemoryScope:
        legal = {scope.key: scope for scope in episode.legal_scope_candidates()}
        suggested = [scope for scope in proposal.suggested_scope_refs if scope.key in legal]
        principal = scope_from_external("user", archive.user_id)
        if proposal.memory_type in {"profile", "preference"}:
            selected = [principal, *[scope for scope in suggested if scope.kind in {"workspace", "environment"}]]
        elif suggested:
            selected = suggested
        elif episode.origin.primary_scope is not None:
            selected = [episode.origin.primary_scope]
        else:
            selected = [principal]
        return MemoryScope(
            applicability=ScopeSelector(tuple({scope.key: scope for scope in selected}.values())),
            visibility=VisibilityPolicy(
                episode.tenant_id,
                allowed_principal_ids=(archive.user_id,),
            ),
            origin_refs=episode.origin.scope_refs,
        )

    def _compatibility_metadata(
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
            metadata["admission"] = {"decision": "accept", "confidence": proposal.confidence}
            metadata["extractor_version"] = proposal.extractor_version
            metadata["model_id"] = proposal.model_id
            metadata["prompt_version"] = proposal.prompt_version
            metadata["identity_fields"] = dict(proposal.identity_fields)
            metadata["merge_key"] = str(metadata.get("claim_id") or metadata.get("slot_id") or "")
            metadata["source_adapter_id"] = str(proposal.metadata.get("source_adapter_id", ""))
            metadata["source_session_id"] = str(proposal.metadata.get("source_session_id", ""))
            metadata["source_roles"] = [str(proposal.metadata.get("source_role", "user"))]
            metadata["source"] = {
                "adapter_id": metadata["source_adapter_id"],
                "session_id": metadata["source_session_id"],
                "roles": metadata["source_roles"],
            }
            object_payload["metadata"] = metadata
            operation.payload.update(
                {
                    "memory_type": proposal.memory_type,
                    "admission": metadata["admission"],
                    "retrieval_views": list(retrieval_views),
                    "schema_version": "canonical_memory_v1",
                    "source_adapter_id": metadata["source_adapter_id"],
                    "source_session_id": metadata["source_session_id"],
                    "source_roles": metadata["source_roles"],
                    "merge_key": metadata["merge_key"],
                    "merge_decision": "ADD" if operation.payload.get("expected_revision", 0) == 0 else "UPDATE",
                    "existing_uri": operation.target_uri if operation.payload.get("expected_revision", 0) != 0 else "",
                    "merge_reason": "canonical_transition_policy",
                }
            )
