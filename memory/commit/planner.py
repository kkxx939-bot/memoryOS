"""从不可变 Session 证据生成确定性的 Markdown Memory 编辑计划。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from difflib import unified_diff
from pathlib import Path
from typing import Any

from foundation.ids import require_safe_path_segment
from foundation.integrity import canonical_digest
from infrastructure.store.contracts.session_archive import SessionArchiveStore
from infrastructure.store.memory.bootstrap import MemoryDocumentBootstrapper
from infrastructure.store.memory.evidence import (
    DurableSalienceLedger,
    SealedProposalSet,
    SealedProposalStore,
)
from infrastructure.store.memory.review import MemoryEditReviewStore
from memory.core.formation import EpisodeSalienceGate, MemoryCandidateRegistry, RuleFallbackExtractor
from memory.core.model import DocumentEditPlan, MemoryEditProposal, PresentPath
from memory.execute.write_planner import MemoryDocumentPlanner
from memory.formation import MemoryExtractionError, classify_memory_extraction_failure
from memory.ports import MemoryExtractorBackend
from pre.evidence import SessionArchiveEpisodeAdapter
from pre.session import SessionArchive


class MemoryExtractionBackendError(RuntimeError):
    def __init__(self, error_type: str, *, retryable: bool) -> None:
        self.error_type = error_type
        self.retryable = retryable
        super().__init__(f"memory extraction backend failed: {error_type}")


@dataclass(frozen=True)
class PlannedMemoryEdit:
    proposal: MemoryEditProposal
    plan: DocumentEditPlan


@dataclass(frozen=True)
class MemoryDocumentPlanningResult:
    edits: tuple[PlannedMemoryEdit, ...]
    proposal_set_digest: str
    edit_proposal_count: int
    edit_proposal_ids: tuple[str, ...] = ()
    candidate_count: int = 0
    salience_reasons: tuple[str, ...] = ()
    egress_decision: str = "LOCAL_ONLY"


class MemoryCommitPlanner:
    """模型最多调用一次，之后只执行确定性的路由和编辑规划。"""

    def __init__(
        self,
        document_planner: MemoryDocumentPlanner,
        *,
        extractor: MemoryExtractorBackend | None = None,
        registry: MemoryCandidateRegistry | None = None,
        archive_store: SessionArchiveStore | None = None,
        proposal_store: SealedProposalStore | None = None,
        salience_gate: EpisodeSalienceGate | None = None,
        salience_ledger: DurableSalienceLedger | None = None,
        bootstrapper: MemoryDocumentBootstrapper | None = None,
        review_store: MemoryEditReviewStore | None = None,
        auto_review_confidence_threshold: float = 0.75,
        root: str | Path | None = None,
        tenant_id: str = "default",
    ) -> None:
        if not 0.0 <= float(auto_review_confidence_threshold) <= 1.0:
            raise ValueError("auto review confidence threshold must be between zero and one")
        self.document_planner = document_planner
        self.extractor = extractor or RuleFallbackExtractor()
        self.registry = registry or MemoryCandidateRegistry()
        self.archive_store = archive_store
        self.salience_gate = salience_gate or EpisodeSalienceGate()
        self.salience_ledger = salience_ledger
        self.bootstrapper = bootstrapper
        self.auto_review_confidence_threshold = float(auto_review_confidence_threshold)
        self.tenant_id = require_safe_path_segment(tenant_id, "memory planner tenant_id")
        self.proposal_store: SealedProposalStore | None = proposal_store
        if self.proposal_store is None and root is not None:
            self.proposal_store = SealedProposalStore(root, tenant_id=self.tenant_id)
        self.review_store = review_store
        if self.review_store is None and root is not None:
            self.review_store = MemoryEditReviewStore(root)

    def plan_session(
        self,
        archive: SessionArchive,
        *,
        tenant_id: str,
        owner_user_id: str,
        commit_group_id: str,
    ) -> MemoryDocumentPlanningResult:
        tenant = require_safe_path_segment(tenant_id, "memory plan tenant_id")
        owner = require_safe_path_segment(owner_user_id, "memory plan owner_user_id")
        if tenant != self.tenant_id or owner != archive.user_id:
            raise ValueError("memory plan crosses trusted tenant or owner boundary")
        if not commit_group_id:
            raise ValueError("memory plan commit_group_id is required")
        if self.proposal_store is not None:
            self.proposal_store.assert_task_replay_allowed(owner, archive.task_id)
        if self.bootstrapper is not None:
            self.bootstrapper.ensure_user(tenant, owner)
        persisted = self._persisted_archive(archive, tenant)
        sealed, reasons, egress = self._sealed_proposals(persisted, tenant, owner)
        edits: list[PlannedMemoryEdit] = []
        review_ids: list[str] = []
        planned_bindings: list[DocumentEditPlan] = []
        for index, proposal in enumerate(sealed.proposals):
            plan = self.document_planner.plan(
                proposal,
                tenant_id=tenant,
                owner_user_id=owner,
                idempotency_key=f"{commit_group_id}:memory:{index}:{sealed.proposal_set_digest}",
                evidence_digest=self._proposal_evidence_digest(persisted, proposal),
            )
            planned_bindings.append(plan)
            if float(proposal.confidence) < self.auto_review_confidence_threshold:
                proposed_diff = self._proposal_diff(plan)
                # 精确 no-op 计划不会修改 Source，因此不需要人工授权；
                # 它继续走幂等直通路径，让 Session 消费者能够正常结束。
                if proposed_diff:
                    if self.review_store is None:
                        raise RuntimeError("uncertain automatic memory requires a durable review store")
                    review = self.review_store.seal(
                        plan,
                        proposed_diff=proposed_diff,
                        independent_evidence_references=(persisted.archive_uri,),
                    )
                    review_ids.append(review.proposal_id)
                    continue
            edits.append(PlannedMemoryEdit(proposal=proposal, plan=plan))
        if planned_bindings and self.proposal_store is not None:
            self.proposal_store.bind_documents(
                task_id=persisted.task_id,
                owner_user_id=owner,
                proposal_set_digest=sealed.proposal_set_digest,
                document_bindings=tuple(
                    (plan.document_id, self._plan_change_digest(plan)) for plan in planned_bindings
                ),
            )
        return MemoryDocumentPlanningResult(
            edits=tuple(edits),
            proposal_set_digest=sealed.proposal_set_digest,
            edit_proposal_count=len(review_ids),
            edit_proposal_ids=tuple(review_ids),
            candidate_count=len(sealed.proposals),
            salience_reasons=reasons,
            egress_decision=egress,
        )

    def plan(
        self,
        archive: SessionArchive,
        *,
        tenant_id: str,
        owner_user_id: str,
        commit_group_id: str,
    ) -> MemoryDocumentPlanningResult:
        return self.plan_session(
            archive,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
            commit_group_id=commit_group_id,
        )

    def _persisted_archive(self, archive: SessionArchive, tenant_id: str) -> SessionArchive:
        if not archive.archive_digest or not archive.manifest_digest or not archive.archive_uri:
            raise ValueError("memory planning requires a durable SessionArchive and manifest")
        if self.archive_store is None:
            return archive
        persisted = self.archive_store.read_archive_at_manifest(
            archive.archive_uri,
            archive.manifest_digest,
            tenant_id=tenant_id,
        )
        actual = (
            persisted.task_id,
            persisted.session_id,
            persisted.user_id,
            persisted.archive_uri,
            persisted.archive_digest,
            persisted.manifest_digest,
        )
        expected = (
            archive.task_id,
            archive.session_id,
            archive.user_id,
            archive.archive_uri,
            archive.archive_digest,
            archive.manifest_digest,
        )
        if actual != expected:
            raise ValueError("memory planning archive differs from immutable evidence")
        return persisted

    def _sealed_proposals(
        self,
        archive: SessionArchive,
        tenant_id: str,
        owner_user_id: str,
    ) -> tuple[SealedProposalSet, tuple[str, ...], str]:
        if self.proposal_store is not None:
            path = self.proposal_store.path(owner_user_id, archive.task_id)
            if path.exists() or path.is_symlink():
                sealed = self.proposal_store.load(owner_user_id, archive.task_id)
                self._assert_sealed_archive(sealed, archive)
                return sealed, ("replayed_sealed_proposal_set",), "SEALED_REPLAY"
        episode = SessionArchiveEpisodeAdapter().adapt(archive)
        policy = dict(archive.metadata.get("memory_planning", {}) or {})
        project_id = str(archive.metadata.get("project_id") or archive.metadata.get("workspace_id") or "__unscoped__")
        if self.salience_ledger is not None:
            salience = self.salience_ledger.reserve(
                self.salience_gate,
                episode,
                task_id=archive.task_id,
                user_id=owner_user_id,
                project_id=project_id,
                policy_seen_fingerprints=tuple(policy.get("seen_episode_fingerprints", []) or []),
                prior_episode_counts=dict(policy.get("prior_episode_counts", {}) or {}),
                policy_consumed_budget=int(policy.get("consumed_budget", 0) or 0),
                max_episode_budget=int(policy.get("max_episode_budget", 8) or 8),
            ).decision
        else:
            salience = self.salience_gate.evaluate(
                episode,
                seen_episode_fingerprints=tuple(policy.get("seen_episode_fingerprints", []) or []),
                prior_episode_counts=dict(policy.get("prior_episode_counts", {}) or {}),
                consumed_budget=int(policy.get("consumed_budget", 0) or 0),
                max_episode_budget=int(policy.get("max_episode_budget", 8) or 8),
            )
        proposals: tuple[MemoryEditProposal, ...] = ()
        egress = "SKIPPED_LOW_SALIENCE"
        if salience.salient:
            try:
                batch_method = getattr(self.extractor, "extract_batch", None)
                if callable(batch_method):
                    batch: Any = batch_method(archive, self.registry.list())
                    raw = tuple(batch.accepted)
                    egress = str(getattr(batch, "egress_decision", "UNKNOWN"))
                else:
                    raw = tuple(self.extractor.extract(archive, self.registry.list()))
                    egress = "LOCAL_ONLY" if not getattr(self.extractor, "is_remote", True) else "ALLOW"
            except Exception as exc:
                failure = classify_memory_extraction_failure(exc)
                raise MemoryExtractionBackendError(failure.code, retryable=failure.retryable) from exc
            proposals = self._validate_proposals(raw, episode.event_ids)
        if self.proposal_store is None:
            sealed = SealedProposalSet(
                task_id=archive.task_id,
                tenant_id=tenant_id,
                owner_user_id=owner_user_id,
                archive_uri=archive.archive_uri,
                archive_digest=archive.archive_digest,
                manifest_digest=archive.manifest_digest,
                proposals=proposals,
                proposal_set_digest=self._proposal_set_digest(proposals),
            )
        else:
            sealed = self.proposal_store.seal(
                task_id=archive.task_id,
                owner_user_id=owner_user_id,
                archive_uri=archive.archive_uri,
                archive_digest=archive.archive_digest,
                manifest_digest=archive.manifest_digest,
                proposals=proposals,
            )
        return sealed, salience.reasons, egress

    @staticmethod
    def _validate_proposals(proposals: tuple[Any, ...], event_ids: frozenset[str]) -> tuple[MemoryEditProposal, ...]:
        result: list[MemoryEditProposal] = []
        for item in proposals:
            if not isinstance(item, MemoryEditProposal):
                raise MemoryExtractionError("extractor returned a non-semantic proposal")
            if not set(item.evidence_refs).issubset(event_ids):
                raise MemoryExtractionError("extractor proposal references unknown evidence")
            result.append(item)
        unique = {MemoryCommitPlanner._proposal_set_digest((item,)): item for item in result}
        return tuple(unique[key] for key in sorted(unique))

    @staticmethod
    def _proposal_set_digest(proposals: tuple[MemoryEditProposal, ...]) -> str:
        return canonical_digest([item.to_dict() for item in proposals])

    @staticmethod
    def _proposal_evidence_digest(archive: SessionArchive, proposal: MemoryEditProposal) -> str:
        return canonical_digest(
            {
                "archive_uri": archive.archive_uri,
                "archive_digest": archive.archive_digest,
                "manifest_digest": archive.manifest_digest,
                "event_ids": list(proposal.evidence_refs),
            }
        )

    @staticmethod
    def _plan_change_digest(plan: DocumentEditPlan) -> str:
        """只计算规划副作用指纹，不持久化 Markdown 正文字节。"""

        return canonical_digest(
            {
                "document_id": plan.document_id,
                "edit_kind": plan.edit_kind.value,
                "relative_path": plan.relative_path,
                "new_relative_path": plan.new_relative_path,
                "evidence_digest": plan.evidence_digest,
                "idempotency_digest": hashlib.sha256(plan.idempotency_key.encode("utf-8")).hexdigest(),
                "after_digest": (hashlib.sha256(plan.after_bytes).hexdigest() if plan.after_bytes is not None else ""),
            }
        )

    def _proposal_diff(self, plan: DocumentEditPlan) -> bytes:
        if plan.after_bytes is None:
            raise ValueError("automatic review currently requires an exact create/update body")
        before = b""
        if isinstance(plan.expected_state, PresentPath):
            before = self.document_planner.store.read_raw(
                plan.tenant_id,
                plan.owner_user_id,
                relative_path=plan.relative_path,
            )
            if hashlib.sha256(before).hexdigest() != plan.expected_state.raw_sha256:
                raise RuntimeError("review source changed after deterministic planning")
        if before == plan.after_bytes:
            return b""
        try:
            before_lines = before.decode("utf-8", errors="strict").splitlines(keepends=True)
            after_lines = plan.after_bytes.decode("utf-8", errors="strict").splitlines(keepends=True)
        except UnicodeDecodeError as exc:
            raise ValueError("review proposal Markdown must be UTF-8") from exc
        diff = "".join(
            unified_diff(
                before_lines,
                after_lines,
                fromfile="live-markdown",
                tofile="proposed-markdown",
            )
        ).encode("utf-8")
        if not diff:
            raise RuntimeError("changed review proposal produced an empty bounded diff")
        return diff

    @staticmethod
    def _assert_sealed_archive(sealed: SealedProposalSet, archive: SessionArchive) -> None:
        actual = (
            sealed.task_id,
            sealed.owner_user_id,
            sealed.archive_uri,
            sealed.archive_digest,
            sealed.manifest_digest,
        )
        expected = (
            archive.task_id,
            archive.user_id,
            archive.archive_uri,
            archive.archive_digest,
            archive.manifest_digest,
        )
        if actual != expected:
            raise ValueError("sealed proposal set crosses immutable archive boundary")


__all__ = [
    "MemoryCommitPlanner",
    "MemoryDocumentPlanningResult",
    "MemoryExtractionBackendError",
    "PlannedMemoryEdit",
]
