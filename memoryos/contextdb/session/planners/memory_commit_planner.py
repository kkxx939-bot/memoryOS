"""上下文数据库里的记忆提交规划器。"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from typing import Any, cast

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
from memoryos.memory.canonical import (
    AliasRegistry,
    CanonicalMemoryFormationService,
    EpisodeSalienceGate,
    EvidenceRef,
    ExistingMemoryPrefetcher,
    MemorySemanticProposal,
    SessionArchiveEpisodeAdapter,
)
from memoryos.memory.extraction import MemoryExtractionBatchResult, MemoryExtractorBackend
from memoryos.memory.schema import (
    MemoryOperationGroup,
    MemoryType,
    MemoryTypeRegistry,
)
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
        admission_gate: Any | None = None,
        view_router: MemoryViewRouter | None = None,
        source_store: SourceStore | None = None,
        index_store: IndexStore | None = None,
        relation_store: RelationStore | None = None,
        hybrid_search: HybridSearch | None = None,
        alias_registry: AliasRegistry | None = None,
    ) -> None:
        self.registry = registry or MemoryTypeRegistry()
        self.extractor = extractor
        self.view_router = view_router or getattr(admission_gate, "view_router", None) or MemoryViewRouter()
        self.admission_gate = admission_gate
        self.episode_adapter = SessionArchiveEpisodeAdapter()
        self.salience_gate = EpisodeSalienceGate()
        self.prefetcher = ExistingMemoryPrefetcher(
            source_store,
            index_store,
            relation_store,
            hybrid_search=hybrid_search,
        )
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
        project_id = project_id_from_archive(archive)
        adapter_id = adapter_id_from_archive(archive)
        batch_extract = getattr(self.extractor, "extract_batch_with_context", None)
        contextual_extract = getattr(self.extractor, "extract_with_context", None)
        extraction_security_flags: tuple[str, ...] = ()
        extracted: Any = []
        try:
            if self.extractor is None:
                batch_result = None
            else:
                batch_result = cast(
                    MemoryExtractionBatchResult | None,
                    batch_extract(
                        archive,
                        schemas,
                        existing_memories=prefetch,
                        episode=episode,
                    )
                    if callable(batch_extract)
                    else None,
                )
            if self.extractor is not None and batch_result is not None:
                extracted = list(batch_result.accepted)
                extraction_security_flags = tuple(batch_result.security_flags)
                outcomes.extend(
                    ProposalPlanningOutcome(
                        item.proposal_id or f"rejected_candidate_{item.index}",
                        "REJECT",
                        item.reason,
                        candidate_index=item.index,
                        security_flags=tuple(item.security_flags),
                    )
                    for item in batch_result.rejected
                )
            elif self.extractor is not None:
                extracted = (
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
            if not isinstance(candidate, MemorySemanticProposal):
                raise TypeError("memory extractor must emit MemorySemanticProposal objects")
            semantic_proposals.append(candidate)
        for proposal in semantic_proposals:
            proposal_views = self._proposal_views(proposal, archive.user_id, project_id, adapter_id)
            canonical_inputs.append(ProposalPlanningInput(proposal, tuple(proposal_views)))
            evidence_refs.extend(proposal.evidence_refs)
            formed = self.formation.plan(
                proposal,
                archive=archive,
                episode=episode,
                retrieval_views=proposal_views,
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
            extraction_security_flags,
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
        outcomes: list[ProposalPlanningOutcome] = [
            item for item in context.proposal_outcomes if item.decision == "REJECT"
        ]
        for item in inputs:
            if item.forced_pending_reason:
                formed = self.formation.plan_pending(
                    item.proposal,
                    archive=archive,
                    episode=episode,
                    reason=item.forced_pending_reason,
                    retrieval_views=list(item.retrieval_views),
                    commit_group_id=context.operation_group_identity,
                )
            else:
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
            context.extraction_security_flags,
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
        extraction_security_flags: tuple[str, ...] = (),
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
            extraction_security_flags=tuple(extraction_security_flags),
            salience_fingerprint=salience_fingerprint,
            salience_reasons=salience_reasons,
        )

    def _proposal_views(
        self,
        proposal: MemorySemanticProposal,
        user_id: str,
        project_id: str,
        adapter_id: str,
    ) -> list[str]:
        schema = self.registry.get(MemoryType(proposal.memory_type))
        return self.view_router.route(
            proposal,
            schema,
            user_id=user_id,
            project_id=project_id,
            adapter_id=adapter_id,
        )

MemoryCommitPlanner = RuleMemoryCommitPlanner
