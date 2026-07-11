"""上下文数据库里的记忆提交规划器。"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from typing import Any

from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
from memoryos.contextdb.session.planning import (
    MemoryPlanningResult,
    PlanningContext,
    PrefetchSnapshot,
    ProposalPlanningInput,
    ProposalPlanningOutcome,
    StagedObjectSnapshot,
)
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.source_store import IndexStore, RelationStore, SourceStore
from memoryos.core.ids import stable_hash
from memoryos.memory.admission import MemoryAdmissionGate
from memoryos.memory.canonical import (
    AliasRegistry,
    CandidateProposalAdapter,
    CanonicalMemoryFormationService,
    EpisodeSalienceGate,
    EvidenceRef,
    ExistingMemoryPrefetcher,
    MemorySemanticProposal,
    SessionArchiveEpisodeAdapter,
)
from memoryos.memory.extraction import MemoryExtractorBackend, RuleFallbackExtractor
from memoryos.memory.model.memory import MemoryCandidate, MemoryKind
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


class MemoryExtractionBackendError(RuntimeError):
    def __init__(self, error_type: str, *, retryable: bool) -> None:
        self.error_type = error_type
        self.retryable = retryable
        super().__init__(f"memory extraction backend failed: {error_type}")


class RuleMemoryCommitPlanner:
    """把会话证据整理成规范记忆事务。"""

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
        self.candidate_adapter = CandidateProposalAdapter()
        self.formation = CanonicalMemoryFormationService(source_store, alias_registry=alias_registry)

    def plan(self, archive: SessionArchive) -> MemoryPlanningResult:
        """处理 plan 这一步。"""

        operations: list[ContextOperation] = []
        group = MemoryOperationGroup()
        schemas = self.registry.list()
        staging: dict[str, Any] = {}
        canonical_inputs: list[ProposalPlanningInput] = []
        outcomes: list[ProposalPlanningOutcome] = []
        evidence_refs: list[EvidenceRef] = []
        episode = self.episode_adapter.adapt(archive)
        prefetch = self.prefetcher.prefetch(episode, owner_user_id=archive.user_id)
        planning_policy = dict(archive.metadata.get("memory_planning", {}) or {})
        salience = self.salience_gate.evaluate(
            episode,
            existing_memories=prefetch,
            seen_episode_fingerprints=tuple(
                str(item) for item in planning_policy.get("seen_episode_fingerprints", []) or []
            ),
            prior_episode_counts={
                str(key): int(value)
                for key, value in dict(planning_policy.get("prior_episode_counts", {}) or {}).items()
            },
            consumed_budget=int(planning_policy.get("consumed_budget", 0) or 0),
            max_episode_budget=int(planning_policy.get("max_episode_budget", 8) or 8),
        )
        archive_digest = str(getattr(archive, "archive_digest", "") or "")
        manifest_digest = str(getattr(archive, "manifest_digest", "") or "")
        planning_id = stable_hash([archive.task_id, archive.session_id, archive_digest], length=32)
        operation_group_identity = f"commit_group_{archive.task_id}"
        if not salience.salient:
            context = self._context(
                planning_id,
                operation_group_identity,
                archive,
                episode,
                prefetch,
                canonical_inputs,
                staging,
                evidence_refs,
                group,
                operations,
                outcomes,
                salience.episode_fingerprint,
                salience.reasons,
                archive_digest,
                manifest_digest,
            )
            return MemoryPlanningResult((), context)
        project_id = project_id_from_archive(archive)
        adapter_id = adapter_id_from_archive(archive)
        contextual_extract = getattr(self.extractor, "extract_with_context", None)
        try:
            extracted: Any = (
                contextual_extract(
                    archive,
                    schemas,
                    existing_memories=prefetch,
                    episode=episode,
                )
                if callable(contextual_extract)
                else self.extractor.extract(archive, schemas)
            )
        except (ConnectionError, OSError, RuntimeError, TimeoutError) as exc:
            raise MemoryExtractionBackendError(type(exc).__name__, retryable=True) from exc
        semantic_proposals: list[MemorySemanticProposal] = []
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
            canonical_inputs.append(ProposalPlanningInput(proposal, tuple(item.admission.retrieval_views)))
            evidence_refs.extend(proposal.evidence_refs)
            formed = self.formation.plan(
                proposal,
                archive=archive,
                episode=episode,
                retrieval_views=item.admission.retrieval_views,
                staged_objects=staging,
                commit_group_id=operation_group_identity,
            )
            outcomes.append(ProposalPlanningOutcome(proposal.proposal_id, formed.decision.value, formed.reason))
            operations.extend(formed.operations)
            self.formation.stage(formed.operations, staging)
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
            canonical_inputs.append(ProposalPlanningInput(proposal, tuple(views)))
            evidence_refs.extend(proposal.evidence_refs)
            formed = self.formation.plan(
                proposal,
                archive=archive,
                episode=episode,
                retrieval_views=views,
                staged_objects=staging,
                commit_group_id=operation_group_identity,
            )
            outcomes.append(ProposalPlanningOutcome(proposal.proposal_id, formed.decision.value, formed.reason))
            operations.extend(formed.operations)
            self.formation.stage(formed.operations, staging)
        context = self._context(
            planning_id,
            operation_group_identity,
            archive,
            episode,
            prefetch,
            canonical_inputs,
            staging,
            evidence_refs,
            group,
            operations,
            outcomes,
            salience.episode_fingerprint,
            salience.reasons,
            archive_digest,
            manifest_digest,
        )
        return MemoryPlanningResult(tuple(operations), context)

    def replan(self, context: PlanningContext, archive: SessionArchive) -> MemoryPlanningResult:
        """处理 replan last 这一步。"""

        archive_digest = str(getattr(archive, "archive_digest", "") or "")
        manifest_digest = str(getattr(archive, "manifest_digest", "") or "")
        episode = self.episode_adapter.adapt(archive)
        context.assert_matches(
            task_id=archive.task_id,
            session_id=archive.session_id,
            tenant_id=episode.tenant_id,
            archive_digest=archive_digest,
            manifest_digest=manifest_digest,
        )
        inputs = context.proposal_inputs
        staging: dict[str, Any] = {}
        operations: list[ContextOperation] = []
        outcomes: list[ProposalPlanningOutcome] = []
        for item in inputs:
            formed = self.formation.plan(
                item.proposal,
                archive=archive,
                episode=episode,
                retrieval_views=list(item.retrieval_views),
                staged_objects=staging,
                commit_group_id=context.operation_group_identity,
            )
            outcomes.append(ProposalPlanningOutcome(item.proposal.proposal_id, formed.decision.value, formed.reason))
            operations.extend(formed.operations)
            self.formation.stage(formed.operations, staging)
        replanned = self._context(
            context.planning_id,
            context.operation_group_identity,
            archive,
            episode,
            tuple(),
            list(inputs),
            staging,
            [ref for item in inputs for ref in item.proposal.evidence_refs],
            MemoryOperationGroup(),
            operations,
            outcomes,
            context.salience_fingerprint,
            context.salience_reasons,
            archive_digest,
            manifest_digest,
        )
        replanned = replace(replanned, prefetch_snapshot=context.prefetch_snapshot)
        return MemoryPlanningResult(tuple(operations), replanned)

    def _context(
        self,
        planning_id: str,
        group_id: str,
        archive: SessionArchive,
        episode: Any,
        prefetch: tuple[Any, ...],
        inputs: list[ProposalPlanningInput],
        staging: dict[str, Any],
        evidence_refs: list[Any],
        group: MemoryOperationGroup,
        operations: list[ContextOperation],
        outcomes: list[ProposalPlanningOutcome],
        salience_fingerprint: str,
        salience_reasons: tuple[str, ...],
        archive_digest: str,
        manifest_digest: str,
    ) -> PlanningContext:
        snapshots = tuple(
            PrefetchSnapshot(
                uri=str(item.uri),
                revision=int(item.revision),
                payload_json=json.dumps(asdict(item), ensure_ascii=False, sort_keys=True, default=str),
            )
            for item in prefetch
        )
        staged = tuple(
            StagedObjectSnapshot(
                uri=str(uri),
                payload_json=json.dumps(obj.to_dict(), ensure_ascii=False, sort_keys=True, default=str),
            )
            for uri, obj in sorted(staging.items())
        )
        planned = tuple(
            sorted(
                {
                    str(operation.target_uri): int(operation.payload.get("expected_revision", 0))
                    for operation in operations
                    if operation.target_uri and operation.payload.get("canonical_memory") is True
                }.items()
            )
        )
        return PlanningContext(
            planning_id=planning_id,
            task_id=archive.task_id,
            archive_digest=archive_digest,
            manifest_digest=manifest_digest,
            episode_id=episode.episode_id,
            session_id=archive.session_id,
            tenant_id=episode.tenant_id,
            proposal_inputs=tuple(inputs),
            prefetch_snapshot=snapshots,
            planned_against_revisions=planned,
            staged_objects=staged,
            scope_candidates=tuple(scope.key for scope in episode.legal_scope_candidates()),
            evidence_references=tuple(dict.fromkeys(evidence_refs)),
            operation_group_identity=group_id,
            admission_summary=tuple(sorted(group.summary().items())),
            proposal_outcomes=tuple(outcomes),
            salience_fingerprint=salience_fingerprint,
            salience_reasons=salience_reasons,
        )

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
    ) -> MemoryCandidate:
        admission = item.admission
        source: dict[str, Any] = {
            "adapter_id": candidate.source_adapter_id,
            "session_id": candidate.source_session_id or archive.session_id,
            "message_ids": candidate.source_message_ids,
            "roles": [candidate.source_role],
        }
        uri = self._candidate_uri(archive.user_id, candidate)
        title = candidate.title[:96] or candidate.memory_type.value
        if admission.decision != AdmissionDecision.PENDING:
            raise ValueError("non-canonical memory operations are limited to pending candidates")
        confidence = min(candidate.confidence, admission.confidence)
        admission_metadata = admission.to_metadata()
        merge_key = admission.merge_key or candidate.merge_key
        return MemoryCandidate(
            uri=uri,
            user_id=archive.user_id,
            title=title,
            content=candidate.content,
            kind=MemoryKind.CANDIDATE,
            confidence=confidence,
            memory_type=candidate.memory_type.value,
            retrieval_views=admission.retrieval_views,
            admission=admission_metadata,
            merge_key=merge_key,
            fields=dict(candidate.fields),
            source=source,
            memory_schema_version=MEMORY_SCHEMA_VERSION,
        )

    def _candidate_uri(self, user_id: str, candidate: MemoryCandidateDraft) -> str:
        digest = stable_hash([user_id, candidate.memory_type.value, candidate.merge_key], length=20)
        return f"memoryos://user/{user_id}/memories/candidates/{digest}"

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
