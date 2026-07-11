"""记忆系统里的形成。"""

from __future__ import annotations

import re
from collections.abc import Mapping
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
from memoryos.memory.canonical.event import canonicalize
from memoryos.memory.canonical.evidence import EvidenceRef, ProposalEvidenceValidator, bind_field_evidence
from memoryos.memory.canonical.identity import (
    AliasRegistry,
    StableMemoryIdentityResolver,
)
from memoryos.memory.canonical.proposal import (
    EpistemicStatus,
    MemorySemanticProposal,
    SemanticAssessment,
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
from memoryos.memory.canonical.semantic import MemorySemanticNormalizer
from memoryos.memory.canonical.state import MemoryClaim, MemorySlot
from memoryos.memory.canonical.transaction import MemoryTransactionPlanner
from memoryos.memory.canonical.transition import MemoryTransitionPolicy, PendingSemanticReconciliation
from memoryos.memory.schema import MemoryCandidateDraft, MemoryType
from memoryos.operations.model.context_operation import ContextOperation


@dataclass(frozen=True)
class CanonicalFormationResult:
    """保存 CanonicalFormationResult 需要的这组数据。"""

    operations: tuple[ContextOperation, ...]
    decision: ProposalAdmissionDecision
    reason: str
    proposal: MemorySemanticProposal


class CandidateProposalAdapter:
    """Convert a typed candidate draft into an evidence-bound proposal."""

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
            related_memory_ids=(),
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
            },
        )

    def _identity(self, candidate: MemoryCandidateDraft, episode: EvidenceEpisode) -> tuple[dict[str, Any], list[str]]:
        fields = candidate.fields
        topic = self._topic(candidate.content)
        if candidate.memory_type == MemoryType.PROFILE:
            explicit_key = fields.get("attribute_key")
            return {"attribute_key": str(explicit_key or topic)}, ["attribute_key"] if explicit_key else []
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
            if field_name
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
        if candidate.memory_type == MemoryType.PROFILE:
            values["canonical_value"] = candidate.content
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
                    and key == "canonical_value"
                    and str(value).casefold() in {"forbidden", "required"}
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

    def __init__(self, source_store: SourceStore | None, *, alias_registry: AliasRegistry | None = None) -> None:
        self.source_store = source_store
        self.validator = ProposalEvidenceValidator()
        self.normalizer = MemorySemanticNormalizer()
        self.admission = ProposalAdmissionGate()
        self.identity = StableMemoryIdentityResolver(alias_registry)
        self.reconciler = MemorySemanticReconciler()
        self.transition = MemoryTransitionPolicy()
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
        """处理 plan 这一步。"""

        proposal = self._bind_evidence_context(self._bind_system_identity(proposal, episode))
        memory_scope = self._memory_scope(proposal, archive, episode)
        raw_validation = self.validator.validate(proposal, episode)
        normalized = self.normalizer.normalize(raw_validation.proposal)
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
        slot: MemorySlot | None = None
        claims: tuple[MemoryClaim, ...] = ()
        if self.source_store is not None or staged_objects:
            repository = CanonicalMemoryRepository(cast(SourceStore, self._planning_source(staged_objects)))
            slot, claims = repository.load(identity)
        reconciled = self.reconciler.reconcile(normalized, identity, slot=slot, claims=claims)
        try:
            transition = self.transition.apply(normalized, identity, reconciled)
        except PendingSemanticReconciliation as pending:
            return CanonicalFormationResult(
                (),
                ProposalAdmissionDecision.PENDING,
                f"semantic_reconciliation_pending:{pending.reason}",
                normalized,
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

    def _planning_source(self, staged_objects: Mapping[str, ContextObject] | None = None):  # noqa: ANN202
        request_staging = dict(staged_objects or {})
        service = self

        class PlanningSource:
            def read_object(self, uri: str) -> ContextObject:
                if uri in request_staging:
                    return request_staging[uri]
                if service.source_store is None:
                    raise FileNotFoundError(uri)
                return service.source_store.read_object(uri)

            def list_objects(self) -> list[ContextObject]:
                persisted = service.source_store.list_objects() if service.source_store is not None else []
                merged = {obj.uri: obj for obj in persisted}
                merged.update(request_staging)
                return [merged[uri] for uri in sorted(merged)]

        return PlanningSource()

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
            metadata["admission"] = {"decision": "accept", "confidence": proposal.confidence}
            metadata["extractor_version"] = proposal.extractor_version
            metadata["model_id"] = proposal.model_id
            metadata["prompt_version"] = proposal.prompt_version
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
