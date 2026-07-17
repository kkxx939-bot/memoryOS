"""Canonical Memory planning for committed sessions."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

from memoryos.contextdb.retrieval.hybrid_search import HybridSearch
from memoryos.contextdb.session.archive_store import SessionArchiveStore
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.index_store import IndexStore
from memoryos.contextdb.store.relation_store import RelationStore
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.core.clock import utc_now
from memoryos.core.ids import stable_hash
from memoryos.core.integrity import canonical_digest
from memoryos.memory.canonical import (
    AliasRegistry,
    CanonicalMemoryFormationService,
    EpisodeSalienceGate,
    EvidenceRef,
    ExistingMemoryPrefetcher,
    MemorySemanticProposal,
    SessionArchiveEpisodeAdapter,
)
from memoryos.memory.canonical.salience_ledger import DurableSalienceLedger
from memoryos.memory.extraction import (
    EgressDecision,
    LLMMemoryExtractorBackend,
    MemoryEgressPolicy,
    MemoryExtractionBatchResult,
    MemoryExtractionConfigurationError,
    MemoryExtractionError,
    MemoryExtractorBackend,
)
from memoryos.memory.extraction.errors import classify_memory_extraction_failure
from memoryos.memory.integration.archive_reader import session_evidence_archive_reader
from memoryos.memory.integration.planning_context import (
    MemoryPlanningResult,
    PlanningContext,
    PrefetchSnapshot,
    ProposalPlanningInput,
    ProposalPlanningOutcome,
    StagedObjectSnapshot,
)
from memoryos.memory.integration.planning_envelope import (
    PlanningEnvelopeIntegrityError,
    PlanningEnvelopeStore,
)
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


def _default_archive_store(root: str | Path, tenant_id: str) -> SessionArchiveStore:
    """Restore the stable direct-construction path for filesystem SourceStore."""

    return cast(SessionArchiveStore, session_evidence_archive_reader(root, tenant_id))


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
        egress_policy: MemoryEgressPolicy | None = None,
        archive_store: SessionArchiveStore | None = None,
    ) -> None:
        self.registry = registry or MemoryTypeRegistry()
        self.extractor = extractor
        self.egress_policy = egress_policy or getattr(extractor, "egress_policy", None) or MemoryEgressPolicy()
        if isinstance(extractor, LLMMemoryExtractorBackend):
            # The planner is the process-wide policy boundary. A configured
            # policy must also govern prompt construction inside the backend;
            # evaluating one policy here and sending under another would make
            # ALLOW_REDACTED and its audit proof meaningless.
            extractor.egress_policy = self.egress_policy
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
        self.formation = CanonicalMemoryFormationService(
            source_store,
            relation_store=relation_store,
            alias_registry=alias_registry,
        )
        root = getattr(source_store, "root", None)
        tenant_id = str(getattr(source_store, "tenant_id", "default"))
        self.planning_store = PlanningEnvelopeStore(root, tenant_id=tenant_id) if root is not None else None
        self.archive_store = (
            archive_store
            if archive_store is not None
            else _default_archive_store(root, tenant_id) if root is not None else None
        )
        if self.archive_store is not None and root is not None:
            if (
                self.archive_store.root.resolve() != Path(root).resolve()
                or self.archive_store.tenant_id != tenant_id
            ):
                raise ValueError("memory planner archive store differs from its SourceStore")
        self.salience_ledger = DurableSalienceLedger(root, tenant_id=tenant_id) if root is not None else None

    def plan(self, archive: SessionArchive) -> MemoryPlanningResult:
        """处理 plan 这一步。"""

        return self.plan_with_progress(archive)

    def plan_with_progress(
        self,
        archive: SessionArchive,
        *,
        progress: Callable[[str, str], None] | None = None,
    ) -> MemoryPlanningResult:
        """Plan while durably reporting the pre-model reservation boundary."""

        if self.planning_store is not None:
            with self.planning_store.task_lock(archive.task_id):
                return self._plan_once(
                    archive,
                    envelope_locked=True,
                    progress=progress,
                )
        return self._plan_once(
            archive,
            envelope_locked=False,
            progress=progress,
        )

    def _plan_once(
        self,
        archive: SessionArchive,
        *,
        envelope_locked: bool,
        progress: Callable[[str, str], None] | None,
    ) -> MemoryPlanningResult:
        archive = self._validated_persisted_archive(archive)
        operations: list[ContextOperation] = []
        group = MemoryOperationGroup()
        schemas = self.registry.list()
        staging: dict[str, Any] = {}
        canonical_inputs: list[ProposalPlanningInput] = []
        outcomes: list[ProposalPlanningOutcome] = []
        evidence_refs: list[EvidenceRef] = []
        episode = self.episode_adapter.adapt(archive)
        archive_digest = str(getattr(archive, "archive_digest", "") or "")
        manifest_digest = str(getattr(archive, "manifest_digest", "") or "")
        if self.planning_store is not None:
            existing = self.planning_store.load(archive.task_id)
            if existing is not None:
                existing.assert_matches(
                    task_id=archive.task_id,
                    session_id=archive.session_id,
                    tenant_id=episode.tenant_id,
                    user_id=archive.user_id,
                    archive_digest=archive_digest,
                    manifest_digest=manifest_digest,
                )
                return self.replan(existing, archive)
        prefetch = self.prefetcher.prefetch(episode, owner_user_id=archive.user_id)
        planning_policy = dict(archive.metadata.get("memory_planning", {}) or {})
        project_id = project_id_from_archive(archive)
        salience_budget_scope = project_id or "__tenant_user_unscoped__"
        policy_seen_fingerprints = tuple(
            str(item) for item in planning_policy.get("seen_episode_fingerprints", []) or []
        )
        prior_episode_counts = {
            str(key): int(value) for key, value in dict(planning_policy.get("prior_episode_counts", {}) or {}).items()
        }
        policy_consumed_budget = int(planning_policy.get("consumed_budget", 0) or 0)
        max_episode_budget = int(planning_policy.get("max_episode_budget", 8) or 8)
        salience_reservation_digest = ""
        if self.salience_ledger is not None:
            reservation = self.salience_ledger.reserve(
                self.salience_gate,
                episode,
                task_id=archive.task_id,
                user_id=archive.user_id,
                project_id=salience_budget_scope,
                existing_memories=prefetch,
                policy_seen_fingerprints=policy_seen_fingerprints,
                prior_episode_counts=prior_episode_counts,
                policy_consumed_budget=policy_consumed_budget,
                max_episode_budget=max_episode_budget,
            )
            salience = reservation.decision
            salience_reservation_digest = reservation.reservation_digest
            if progress is not None:
                progress("salience_reserved", salience_reservation_digest)
            if not reservation.created and salience.salient:
                raise PlanningEnvelopeIntegrityError(
                    "salient extraction was already reserved without a durable planning envelope; "
                    "re-extraction requires a new commit group"
                )
        else:
            salience = self.salience_gate.evaluate(
                episode,
                existing_memories=prefetch,
                seen_episode_fingerprints=policy_seen_fingerprints,
                prior_episode_counts=prior_episode_counts,
                consumed_budget=policy_consumed_budget,
                max_episode_budget=max_episode_budget,
            )
        planning_id = stable_hash([archive.task_id, archive.session_id, archive_digest], length=32)
        operation_group_identity = f"commit_group_{archive.task_id}"
        adapter_id = adapter_id_from_archive(archive)
        batch_extract = getattr(self.extractor, "extract_batch_with_context", None)
        contextual_extract = getattr(self.extractor, "extract_with_context", None)
        extraction_security_flags: tuple[str, ...] = ()
        egress_decision = "LOCAL_ONLY"
        egress_audit: tuple[tuple[str, str], ...] = ()
        extracted: Any = []
        provider = getattr(self.extractor, "provider", self.extractor)
        provider_name = type(provider).__name__ if provider is not None else ""
        model_id = str(getattr(self.extractor, "model_id", "") or "")
        if not salience.salient:
            outcome = "RESTRICTED" if salience.privacy_risk else "ARCHIVE_ONLY"
            skipped_egress_decision = "DENY" if salience.privacy_risk else "LOCAL_ONLY"
            skipped_egress_audit = tuple(
                sorted(
                    {
                        "outbound_digest": "",
                        "decision": skipped_egress_decision,
                        "provider": provider_name,
                        "model": model_id,
                    }.items()
                )
            )
            outcomes.append(
                ProposalPlanningOutcome(
                    proposal_id=f"episode_{episode.episode_id}",
                    decision=outcome,
                    reason=salience.reasons[0] if salience.reasons else "low_salience",
                    security_flags=("privacy_egress_blocked",) if salience.privacy_risk else (),
                )
            )
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
                ("privacy_egress_blocked",) if salience.privacy_risk else (),
                skipped_egress_decision,
                skipped_egress_audit,
            )
            context = replace(
                context,
                salience_score=salience.score,
                salience_budget_cost=salience.budget_cost,
                salience_duplicate=salience.duplicate,
                salience_privacy_risk=salience.privacy_risk,
                salience_reservation_digest=salience_reservation_digest,
                salience_factors=tuple(
                    (factor.name, factor.weight, tuple(factor.event_ids)) for factor in salience.factors
                ),
            )
            return self._seal(context, tuple(), archive, assume_locked=envelope_locked)
        remote_extractor = bool(self.extractor is not None and getattr(self.extractor, "is_remote", True) is not False)
        assessment = self.egress_policy.evaluate(
            archive,
            episode,
            remote=remote_extractor,
            existing_memories=prefetch,
        )
        unsafe_generic_redaction = bool(
            remote_extractor
            and assessment.decision == EgressDecision.ALLOW_REDACTED
            and not isinstance(self.extractor, LLMMemoryExtractorBackend)
        )
        blocked_egress = bool(
            remote_extractor
            and (assessment.decision in {EgressDecision.DENY, EgressDecision.LOCAL_ONLY} or unsafe_generic_redaction)
        )
        boundary_decision = EgressDecision.LOCAL_ONLY.value if unsafe_generic_redaction else assessment.decision.value
        boundary_audit = {
            "outbound_digest": ""
            if blocked_egress or not remote_extractor
            else canonical_digest(
                {
                    "archive_digest": archive_digest,
                    "manifest_digest": manifest_digest,
                    "episode_id": episode.episode_id,
                    "event_digests": [event.digest for event in episode.events],
                }
            ),
            "decision": boundary_decision,
            "provider": provider_name,
            "model": model_id,
        }
        egress_decision = boundary_decision
        egress_audit = tuple(sorted(boundary_audit.items()))
        if blocked_egress:
            flags = (
                ("privacy_egress_blocked",) if assessment.decision == EgressDecision.DENY else ("egress_local_only",)
            )
            outcomes.append(
                ProposalPlanningOutcome(
                    proposal_id=f"episode_{episode.episode_id}",
                    decision="RESTRICTED",
                    reason="remote_egress_policy_blocked_sensitive_archive",
                    security_flags=flags,
                )
            )
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
                flags,
                boundary_decision,
                egress_audit,
            )
            context = replace(
                context,
                salience_score=salience.score,
                salience_budget_cost=salience.budget_cost,
                salience_duplicate=salience.duplicate,
                salience_privacy_risk=salience.privacy_risk,
                salience_reservation_digest=salience_reservation_digest,
                salience_factors=tuple(
                    (factor.name, factor.weight, tuple(factor.event_ids)) for factor in salience.factors
                ),
            )
            return self._seal(context, tuple(), archive, assume_locked=envelope_locked)
        # The durable salience reservation is published before this boundary.
        # Each attempt has finite provider retries.  A later worker may retry
        # only when SessionCommitService recorded a typed transport failure;
        # an abandoned lease or arbitrary direct replay cannot authorize a
        # second model call for the same task.
        extractor_handles_retries = getattr(self.extractor, "handles_retries", False) is True
        max_attempts = 1 if extractor_handles_retries else 3
        batch_result: MemoryExtractionBatchResult | None = None
        for attempt in range(max_attempts):
            typed_error: MemoryExtractionError | None = None
            original_error: BaseException | None = None
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
                break
            except MemoryExtractionError as exc:
                typed_error = exc
                original_error = exc
            except Exception as exc:
                typed_error = classify_memory_extraction_failure(exc)
                original_error = exc
            assert typed_error is not None and original_error is not None
            if typed_error.retryable and attempt + 1 < max_attempts:
                delay = (0.0, 0.05, 0.2)[min(attempt, 2)]
                if delay:
                    time.sleep(delay)
                continue
            # Typed transport failures are retried only inside this bounded
            # planning attempt.  Once those attempts are exhausted no planning
            # envelope exists, so retrying the same commit group would call the
            # model again and could create a different proposal set.
            retryable_after_attempt = False
            raise MemoryExtractionBackendError(
                typed_error.code,
                retryable=retryable_after_attempt,
            ) from original_error

        if batch_result is not None:
            extraction_security_flags = tuple(batch_result.security_flags)
            # The extractor result is semantic model output, not an authority
            # to rewrite the system-owned egress decision or persist arbitrary
            # audit fields.  Only the built-in backend can report the digest
            # of the exact redacted prompt it sent; local and custom extractors
            # retain the planner's content-bound boundary audit.
            if isinstance(self.extractor, LLMMemoryExtractorBackend) and remote_extractor:
                backend_decision = str(getattr(batch_result, "egress_decision", "") or "")
                backend_audit = dict(getattr(batch_result, "egress_audit", {}) or {})
                outbound_digest = str(backend_audit.get("outbound_digest") or "")
                if (
                    backend_decision != boundary_decision
                    or set(backend_audit) != {"outbound_digest", "decision", "provider", "model"}
                    or str(backend_audit.get("decision") or "") != boundary_decision
                    or str(backend_audit.get("provider") or "") != boundary_audit["provider"]
                    or str(backend_audit.get("model") or "") != boundary_audit["model"]
                    or len(outbound_digest) != 64
                    or any(character not in "0123456789abcdef" for character in outbound_digest)
                ):
                    raise MemoryExtractionConfigurationError(
                        "memory extractor returned an invalid egress audit binding"
                    )
                egress_audit = tuple(sorted((key, str(backend_audit[key])) for key in backend_audit))
            outcomes.extend(
                ProposalPlanningOutcome(
                    item.proposal_id or f"rejected_candidate_{item.index}",
                    "RESTRICTED" if item.security_error else "REJECT",
                    item.reason,
                    candidate_index=item.index,
                    security_flags=tuple(item.security_flags),
                )
                for item in batch_result.rejected
            )
        semantic_proposals: list[MemorySemanticProposal] = []
        for candidate in extracted:
            if not isinstance(candidate, MemorySemanticProposal):
                raise TypeError("memory extractor must emit MemorySemanticProposal objects")
            semantic_proposals.append(candidate)
        for proposal in semantic_proposals:
            proposal_views = self._proposal_views(proposal, archive.user_id, project_id, adapter_id)
            formed = self.formation.plan(
                proposal,
                archive=archive,
                episode=episode,
                retrieval_views=proposal_views,
                staged_objects=staging,
                commit_group_id=operation_group_identity,
            )
            canonical_inputs.append(
                ProposalPlanningInput(
                    formed.proposal,
                    tuple(proposal_views),
                    formed.reason if formed.decision.value == "PENDING" else "",
                )
            )
            evidence_refs.extend(formed.proposal.evidence_refs)
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
            egress_decision,
            egress_audit,
        )
        context = replace(
            context,
            salience_score=salience.score,
            salience_budget_cost=salience.budget_cost,
            salience_duplicate=salience.duplicate,
            salience_privacy_risk=salience.privacy_risk,
            salience_reservation_digest=salience_reservation_digest,
            salience_factors=tuple(
                (factor.name, factor.weight, tuple(factor.event_ids)) for factor in salience.factors
            ),
        )
        return self._seal(
            context,
            tuple(operations),
            archive,
            assume_locked=envelope_locked,
        )

    def _validated_persisted_archive(self, archive: SessionArchive) -> SessionArchive:
        """Use only an integrity-checked immutable archive as model evidence.

        A filesystem-backed planner publishes durable planning artifacts and
        may call a remote model.  Letting its caller supply an uncommitted or
        modified in-memory archive would leave the proposal set detached from
        the evidence manifest even if the later envelope itself were valid.
        In-memory/test-only planners without a durable root retain their
        existing pure planning behavior.
        """

        if self.archive_store is None or self.planning_store is None:
            return archive
        tenant_id = self.archive_store.archive_tenant(archive)
        if tenant_id != self.planning_store.tenant_id:
            raise PlanningEnvelopeIntegrityError("memory planning archive crosses the planner tenant boundary")
        try:
            persisted = self.archive_store.read_archive(
                archive.archive_uri,
                tenant_id=tenant_id,
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise PlanningEnvelopeIntegrityError(
                "durable memory planning requires an integrity-checked immutable session archive"
            ) from exc
        evidence_fields = (
            "user_id",
            "session_id",
            "archive_uri",
            "messages",
            "observations",
            "predictions",
            "action_results",
            "feedback",
            "used_contexts",
            "used_skills",
            "tool_results",
            "metadata",
            "task_id",
            "created_at",
            "schema_version",
        )
        requested_evidence = {field_name: getattr(archive, field_name) for field_name in evidence_fields}
        persisted_evidence = {field_name: getattr(persisted, field_name) for field_name in evidence_fields}
        if canonical_digest(requested_evidence) != canonical_digest(persisted_evidence):
            raise PlanningEnvelopeIntegrityError("memory planning input differs from its immutable session archive")
        if (archive.archive_digest and archive.archive_digest != persisted.archive_digest) or (
            archive.manifest_digest and archive.manifest_digest != persisted.manifest_digest
        ):
            raise PlanningEnvelopeIntegrityError("memory planning archive digest binding is inconsistent")
        if not persisted.archive_digest or not persisted.manifest_digest:
            raise PlanningEnvelopeIntegrityError("immutable memory planning archive is missing its content digests")
        return persisted

    def replan(self, context: PlanningContext, archive: SessionArchive) -> MemoryPlanningResult:
        """处理 replan last 这一步。"""

        archive_digest = str(getattr(archive, "archive_digest", "") or "")
        manifest_digest = str(getattr(archive, "manifest_digest", "") or "")
        episode = self.episode_adapter.adapt(archive)
        context.assert_matches(
            task_id=archive.task_id,
            session_id=archive.session_id,
            tenant_id=episode.tenant_id,
            user_id=archive.user_id,
            archive_digest=archive_digest,
            manifest_digest=manifest_digest,
        )
        inputs = context.proposal_inputs
        staging: dict[str, Any] = {}
        operations: list[ContextOperation] = []
        input_ids = {item.proposal.proposal_id for item in inputs}
        outcomes: list[ProposalPlanningOutcome] = [
            item for item in context.proposal_outcomes if item.proposal_id not in input_ids
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
                    staged_objects=staging,
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
            context.egress_decision,
            context.egress_audit,
        )
        replanned = replace(
            replanned,
            prefetch_snapshot=context.prefetch_snapshot,
            proposal_set_digest=context.proposal_set_digest,
            planning_digest=context.planning_digest,
            created_at=context.created_at,
            salience_score=context.salience_score,
            salience_budget_cost=context.salience_budget_cost,
            salience_duplicate=context.salience_duplicate,
            salience_privacy_risk=context.salience_privacy_risk,
            salience_reservation_digest=context.salience_reservation_digest,
            salience_factors=context.salience_factors,
            extractor_version=context.extractor_version,
            model_id=context.model_id,
            prompt_version=context.prompt_version,
            semantic_contract_version=context.semantic_contract_version,
        )
        for operation in operations:
            operation.payload["planning_digest"] = context.planning_digest
            operation.payload["proposal_set_digest"] = context.proposal_set_digest
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
        egress_decision: str = "LOCAL_ONLY",
        egress_audit: tuple[tuple[str, str], ...] = (),
    ) -> PlanningContext:
        snapshots = tuple(
            PrefetchSnapshot(
                uri=str(item.uri),
                revision=int(item.revision),
                object_digest=canonical_digest(asdict(item)),
                content_digest=canonical_digest({"l0": item.l0, "l1": item.l1, "l2": item.l2}),
                relation_digest=canonical_digest(list(item.relations)),
            )
            for item in prefetch
        )
        staged = tuple(
            StagedObjectSnapshot(
                uri=str(uri),
                revision=int(dict(obj.metadata or {}).get("revision", 0)),
                object_digest=canonical_digest(obj.to_dict()),
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
            proposal_set_digest=canonical_digest(
                [
                    {
                        "proposal": item.proposal.to_dict(),
                        "retrieval_views": list(item.retrieval_views),
                        "forced_pending_reason": item.forced_pending_reason,
                    }
                    for item in inputs
                ]
            ),
            egress_decision=egress_decision,
            egress_audit=tuple(egress_audit),
            user_id=archive.user_id,
            extractor_version=self._extractor_identity("extractor_version", default_to_class=True),
            model_id=self._extractor_identity("model_id"),
            prompt_version=self._extractor_identity("prompt_version"),
            semantic_contract_version=self._extractor_identity("semantic_contract_version"),
            created_at=str(archive.created_at or utc_now()),
        )

    def _extractor_identity(self, field_name: str, *, default_to_class: bool = False) -> str:
        if self.extractor is None:
            return "none" if default_to_class else ""
        value = getattr(self.extractor, field_name, None)
        if value is not None and str(value):
            return str(value)
        return type(self.extractor).__name__ if default_to_class else ""

    def _seal(
        self,
        context: PlanningContext,
        operations: tuple[ContextOperation, ...],
        archive: SessionArchive,
        *,
        assume_locked: bool = False,
    ) -> MemoryPlanningResult:
        if self.planning_store is not None:
            payload = self.planning_store.create(
                context,
                archive_uri=archive.archive_uri,
                assume_locked=assume_locked,
            )
            sealed = self.planning_store.load(context.task_id)
            assert sealed is not None
            planning_digest = str(payload["planning_digest"])
            proposal_set_digest = str(payload["proposal_set_digest"])
        else:
            proposal_set_digest = context.proposal_set_digest
            planning_digest = canonical_digest(
                {
                    "schema_version": "memory_planning_envelope_ephemeral_v1",
                    "task_id": context.task_id,
                    "tenant_id": context.tenant_id,
                    "archive_digest": context.archive_digest,
                    "manifest_digest": context.manifest_digest,
                    "proposal_set_digest": proposal_set_digest,
                    "salience_fingerprint": context.salience_fingerprint,
                    "egress_decision": context.egress_decision,
                    "egress_audit": dict(context.egress_audit),
                }
            )
            sealed = replace(
                context,
                planning_digest=planning_digest,
                proposal_set_digest=proposal_set_digest,
            )
        for operation in operations:
            operation.payload["planning_digest"] = planning_digest
            operation.payload["proposal_set_digest"] = proposal_set_digest
        return MemoryPlanningResult(operations, sealed)

    def bind_runtime_stores(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        relation_store: RelationStore | None,
        *,
        root: Any,
        tenant_id: str,
        archive_store: SessionArchiveStore | None = None,
    ) -> None:
        """Bind a custom planner to the committer's durable runtime boundary.

        SessionCommitService historically accepted a planner constructed with
        only an extractor. Such a planner must not fall back to ephemeral
        planning once it is paired with a real committer.
        """

        existing_source = self.prefetcher.source_store
        existing_index = self.prefetcher.index_store
        existing_relation = self.prefetcher.relation_store
        if existing_source is not None and existing_source is not source_store:
            raise ValueError("memory planner SourceStore differs from its committer")
        if existing_index is not None and existing_index is not index_store:
            raise ValueError("memory planner IndexStore differs from its committer")
        if existing_relation is not None and existing_relation is not relation_store:
            raise ValueError("memory planner RelationStore differs from its committer")
        if self.formation.source_store is not None and self.formation.source_store is not source_store:
            raise ValueError("memory formation SourceStore differs from its committer")
        if self.formation.relation_store is not None and self.formation.relation_store is not relation_store:
            raise ValueError("memory formation RelationStore differs from its committer")

        self.prefetcher.source_store = source_store
        self.prefetcher.index_store = index_store
        self.prefetcher.relation_store = relation_store
        self.formation.source_store = source_store
        self.formation.relation_store = relation_store
        planning_store = PlanningEnvelopeStore(root, tenant_id=tenant_id)
        # A planner already bound from its filesystem SourceStore retains that
        # tenant-scoped archive view.  The service injection fills only the
        # historically unbound custom-planner path.
        bound_archive_store = self.archive_store or archive_store
        if bound_archive_store is None:
            raise RuntimeError("durable memory planning requires an injected SessionArchiveStore")
        salience_ledger = DurableSalienceLedger(root, tenant_id=tenant_id)
        if (
            self.planning_store is not None
            and self.planning_store.artifact_root.resolve() != planning_store.artifact_root.resolve()
        ):
            raise ValueError("memory planner durable root differs from its committer")
        if (
            self.salience_ledger is not None
            and self.salience_ledger.artifact_root.resolve() != salience_ledger.artifact_root.resolve()
        ):
            raise ValueError("memory planner salience root differs from its committer")
        if self.archive_store is not None and (
            self.archive_store.root.resolve() != bound_archive_store.root.resolve()
            or self.archive_store.tenant_id != bound_archive_store.tenant_id
        ):
            raise ValueError("memory planner archive root differs from its committer")
        self.planning_store = planning_store
        self.archive_store = bound_archive_store
        self.salience_ledger = salience_ledger

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
