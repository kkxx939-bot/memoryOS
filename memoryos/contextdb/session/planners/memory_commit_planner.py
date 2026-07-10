from __future__ import annotations

from typing import Any

from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.source_store import IndexStore, RelationStore, SourceStore
from memoryos.core.ids import stable_hash
from memoryos.memory.admission import MemoryAdmissionGate
from memoryos.memory.canonical import (
    AliasRegistry,
    CanonicalMemoryFormationService,
    EpisodeSalienceGate,
    ExistingMemoryPrefetcher,
    LegacyCandidateProposalAdapter,
    MemorySemanticProposal,
    SessionArchiveEpisodeAdapter,
)
from memoryos.memory.extraction import MemoryExtractorBackend, RuleFallbackExtractor
from memoryos.memory.model.memory import Memory, MemoryCandidate, MemoryKind
from memoryos.memory.schema import (
    MEMORY_SCHEMA_VERSION,
    AdmissionDecision,
    MemoryCandidateDraft,
    MemoryOperationGroup,
    MemoryOperationGroupItem,
    MemoryType,
    MemoryTypeRegistry,
)
from memoryos.memory.service.memory_updater import MemoryUpdater
from memoryos.memory.view import MemoryViewRouter, adapter_id_from_archive, project_id_from_archive
from memoryos.operations.model.context_operation import ContextOperation


class RuleMemoryCommitPlanner:
    """Schema-driven memory planner with deterministic fallback extraction."""

    def __init__(
        self,
        extractor: MemoryExtractorBackend | None = None,
        registry: MemoryTypeRegistry | None = None,
        admission_gate: MemoryAdmissionGate | None = None,
        view_router: MemoryViewRouter | None = None,
        source_store: SourceStore | None = None,
        index_store: IndexStore | None = None,
        relation_store: RelationStore | None = None,
        hybrid_search: HybridSearch | None = None,
        alias_registry: AliasRegistry | None = None,
    ) -> None:
        self.registry = registry or MemoryTypeRegistry()
        self.extractor = extractor or RuleFallbackExtractor()
        self.view_router = view_router or getattr(admission_gate, "view_router", None) or MemoryViewRouter()
        self.admission_gate = admission_gate or MemoryAdmissionGate(self.registry, self.view_router)
        self.updater = MemoryUpdater()
        self.episode_adapter = SessionArchiveEpisodeAdapter()
        self.salience_gate = EpisodeSalienceGate()
        self.prefetcher = ExistingMemoryPrefetcher(
            source_store,
            index_store,
            relation_store,
            hybrid_search=hybrid_search,
        )
        self.candidate_adapter = LegacyCandidateProposalAdapter()
        self.formation = CanonicalMemoryFormationService(source_store, alias_registry=alias_registry)
        self.last_prefetch: tuple[Any, ...] = ()
        self.last_canonical_inputs: list[tuple[MemorySemanticProposal, list[str]]] = []
        self.last_group = MemoryOperationGroup()

    def plan(self, archive: SessionArchive) -> list[ContextOperation]:
        operations: list[ContextOperation] = []
        group = MemoryOperationGroup()
        schemas = self.registry.list()
        self.formation.begin_planning()
        self.last_canonical_inputs = []
        episode = self.episode_adapter.adapt(archive)
        if not self.salience_gate.evaluate(episode).salient:
            self.last_group = group
            self.last_prefetch = ()
            return []
        self.last_prefetch = self.prefetcher.prefetch(episode, owner_user_id=archive.user_id)
        project_id = project_id_from_archive(archive)
        adapter_id = adapter_id_from_archive(archive)
        contextual_extract = getattr(self.extractor, "extract_with_context", None)
        extracted: Any = (
            contextual_extract(
                archive,
                schemas,
                existing_memories=self.last_prefetch,
                episode=episode,
            )
            if callable(contextual_extract)
            else self.extractor.extract(archive, schemas)
        )
        semantic_proposals = []
        for candidate in extracted:
            if isinstance(candidate, MemorySemanticProposal):
                semantic_proposals.append(candidate)
                continue
            admission = self.admission_gate.evaluate(
                candidate,
                user_id=archive.user_id,
                project_id=project_id,
                adapter_id=adapter_id,
            )
            group.add(candidate, admission)
        for item in group.accepted:
            proposal = self.candidate_adapter.adapt(item.candidate, episode, archive)
            self.last_canonical_inputs.append((proposal, list(item.admission.retrieval_views)))
            formed = self.formation.plan(
                proposal,
                archive=archive,
                episode=episode,
                retrieval_views=item.admission.retrieval_views,
            )
            operations.extend(formed.operations)
            self.formation.stage(formed.operations)
        for item in group.pending:
            operation = self._operation(archive, item)
            operation.payload.update(
                {
                    "merge_decision": "ADD",
                    "existing_uri": "",
                    "merge_reason": "pending_candidate",
                }
            )
            operations.append(operation)
        for proposal in semantic_proposals:
            views = self._proposal_views(proposal, archive.user_id, project_id)
            self.last_canonical_inputs.append((proposal, views))
            formed = self.formation.plan(
                proposal,
                archive=archive,
                episode=episode,
                retrieval_views=views,
            )
            operations.extend(formed.operations)
            self.formation.stage(formed.operations)
        self.last_group = group
        return operations

    def replan_last(self, archive: SessionArchive) -> list[ContextOperation]:
        episode = self.episode_adapter.adapt(archive)
        inputs = list(self.last_canonical_inputs)
        self.formation.begin_planning()
        operations: list[ContextOperation] = []
        for proposal, views in inputs:
            formed = self.formation.plan(
                proposal,
                archive=archive,
                episode=episode,
                retrieval_views=views,
            )
            operations.extend(formed.operations)
            self.formation.stage(formed.operations)
        return operations

    def _proposal_views(self, proposal: MemorySemanticProposal, user_id: str, project_id: str) -> list[str]:
        if proposal.memory_type == MemoryType.PROFILE.value:
            return [f"user:{user_id}:profile"]
        if proposal.memory_type == MemoryType.PREFERENCE.value:
            return [f"user:{user_id}:preferences"]
        suffix = {
            MemoryType.PROJECT_RULE.value: "rules",
            MemoryType.PROJECT_DECISION.value: "decisions",
            MemoryType.AGENT_EXPERIENCE.value: "agent_experience",
        }.get(proposal.memory_type, "knowledge")
        return [f"project:{project_id}:{suffix}"] if project_id else [f"user:{user_id}:profile"]

    def _operation(self, archive: SessionArchive, item: MemoryOperationGroupItem) -> ContextOperation:
        candidate = item.candidate
        admission = item.admission
        memory = self._memory(archive, candidate, item)
        operation = self.updater.add_memory(memory, evidence=self._evidence(candidate))
        operation.source_session_id = candidate.source_session_id or archive.session_id
        operation.payload = {
            **operation.payload,
            "memory_type": candidate.memory_type.value,
            "admission": admission.to_metadata(),
            "retrieval_views": admission.retrieval_views,
            "source_adapter_id": candidate.source_adapter_id,
            "source_session_id": candidate.source_session_id or archive.session_id,
            "source_roles": [candidate.source_role],
            "merge_key": admission.merge_key or candidate.merge_key,
            "schema_version": MEMORY_SCHEMA_VERSION,
            "fields": dict(candidate.fields),
        }
        context_object = operation.payload.get("context_object")
        if isinstance(context_object, dict):
            metadata = dict(context_object.get("metadata", {}) or {})
            archive_metadata = dict(archive.metadata or {})
            metadata.update(
                {
                    "memory_type": candidate.memory_type.value,
                    "admission": admission.to_metadata(),
                    "retrieval_views": admission.retrieval_views,
                    "source": memory.source,
                    "source_adapter_id": candidate.source_adapter_id,
                    "source_session_id": candidate.source_session_id or archive.session_id,
                    "source_roles": [candidate.source_role],
                    "merge_key": admission.merge_key or candidate.merge_key,
                    "schema_version": MEMORY_SCHEMA_VERSION,
                    "fields": dict(candidate.fields),
                    "connect": dict(archive_metadata.get("connect", {}) or {}),
                    "scope": dict(archive_metadata.get("scope", {}) or {}),
                    "provenance": dict(archive_metadata.get("provenance", {}) or {}),
                }
            )
            context_object["metadata"] = metadata
        return operation

    def _memory(
        self, archive: SessionArchive, candidate: MemoryCandidateDraft, item: MemoryOperationGroupItem
    ) -> Memory:
        admission = item.admission
        source: dict[str, Any] = {
            "adapter_id": candidate.source_adapter_id,
            "session_id": candidate.source_session_id or archive.session_id,
            "message_ids": candidate.source_message_ids,
            "roles": [candidate.source_role],
        }
        uri = self._uri(archive.user_id, candidate, admission.decision)
        title = candidate.title[:96] or candidate.memory_type.value
        kind = self._kind(candidate, admission.decision)
        confidence = min(candidate.confidence, admission.confidence)
        admission_metadata = admission.to_metadata()
        merge_key = admission.merge_key or candidate.merge_key
        if admission.decision == AdmissionDecision.PENDING:
            return MemoryCandidate(
                uri=uri,
                user_id=archive.user_id,
                title=title,
                content=candidate.content,
                kind=kind,
                confidence=confidence,
                memory_type=candidate.memory_type.value,
                retrieval_views=admission.retrieval_views,
                admission=admission_metadata,
                merge_key=merge_key,
                fields=dict(candidate.fields),
                source=source,
                memory_schema_version=MEMORY_SCHEMA_VERSION,
            )
        return Memory(
            uri=uri,
            user_id=archive.user_id,
            title=title,
            content=candidate.content,
            kind=kind,
            confidence=confidence,
            memory_type=candidate.memory_type.value,
            retrieval_views=admission.retrieval_views,
            admission=admission_metadata,
            merge_key=merge_key,
            fields=dict(candidate.fields),
            source=source,
            memory_schema_version=MEMORY_SCHEMA_VERSION,
        )

    def _uri(self, user_id: str, candidate: MemoryCandidateDraft, decision: AdmissionDecision) -> str:
        bucket = "candidates" if decision == AdmissionDecision.PENDING else self._bucket(candidate.memory_type)
        digest = stable_hash([user_id, candidate.memory_type.value, candidate.merge_key, candidate.content], length=20)
        return f"memoryos://user/{user_id}/memories/{bucket}/{digest}"

    def _bucket(self, memory_type: MemoryType) -> str:
        return {
            MemoryType.PROFILE: "profile",
            MemoryType.PREFERENCE: "preferences",
            MemoryType.ENTITY: "entities",
            MemoryType.EVENT: "events",
            MemoryType.PROJECT_RULE: "rules",
            MemoryType.PROJECT_DECISION: "decisions",
            MemoryType.AGENT_EXPERIENCE: "agent_experience",
        }[memory_type]

    def _kind(self, candidate: MemoryCandidateDraft, decision: AdmissionDecision) -> MemoryKind:
        if decision == AdmissionDecision.PENDING:
            return MemoryKind.CANDIDATE
        if candidate.memory_type == MemoryType.PROJECT_RULE:
            return MemoryKind.POLICY
        return MemoryKind.EXPLICIT

    def _evidence(self, candidate: MemoryCandidateDraft) -> list[dict]:
        return [
            *candidate.evidence,
            {
                "source_adapter_id": candidate.source_adapter_id,
                "source_session_id": candidate.source_session_id,
                "source_message_ids": candidate.source_message_ids,
                "reason": candidate.reason,
            },
        ]


MemoryCommitPlanner = RuleMemoryCommitPlanner
