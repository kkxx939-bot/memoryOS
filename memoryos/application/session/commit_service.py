"""Durable greenfield Session commit orchestration.

The immutable SessionArchive is the evidence boundary.  Markdown memory edits
are committed by ``MemoryDocumentCommitter`` and ordinary Session operations
remain on ``OperationCommitter``; neither path delegates durable authority to
the other.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from typing import Any, cast

from memoryos.application.session.commit_entry import commit_session as commit_session_entry
from memoryos.application.session.commit_group import (
    CONSUMERS,
    CommitGroupStatus,
    CommitGroupStore,
    MemoryDocumentEffect,
)
from memoryos.application.session.planners.action_policy_commit_planner import (
    ActionPolicyCommitPlanner,
)
from memoryos.application.session.planners.behavior_commit_planner import BehaviorCommitPlanner
from memoryos.application.session.planners.context_commit_planner import ContextCommitPlanner
from memoryos.application.session.projection_journal import SessionProjectionJournal
from memoryos.contextdb.layers.layer_generator import l0_abstract, l1_overview
from memoryos.contextdb.session.archive_store import SessionArchiveStore
from memoryos.contextdb.session.session_model import (
    SessionArchive,
    SessionCommitResult,
    SessionCommitState,
)
from memoryos.contextdb.store.queue_store import QueueJob, QueueStore
from memoryos.core.errors import RevisionConflictError
from memoryos.core.ids import stable_hash
from memoryos.core.integrity import canonical_digest
from memoryos.memory.documents import (
    DocumentCommitConflict,
    DocumentCommitResult,
    DocumentConflictError,
    DocumentDeletionStatus,
    DocumentEditPlan,
    DocumentIntentStatus,
    MemoryDocumentCommitter,
    MemoryDocumentPlanner,
    MemoryEditProposal,
)
from memoryos.memory.documents.control_store import document_intent_id
from memoryos.memory.documents.layout import tenant_control_root
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation


class DerivedConsumerError(RuntimeError):
    """One or more independent derived consumers did not complete."""

    def __init__(self, failures: Sequence[tuple[str, bool]]) -> None:
        self.failures = tuple(failures)
        self.retryable = bool(self.failures) and all(item[1] for item in self.failures)
        names = ",".join(item[0] for item in self.failures)
        super().__init__(f"Session derived consumers failed: {names}")


class ConsumerLeaseBusy(RuntimeError):
    retryable = True


class ConsumerTerminalError(RuntimeError):
    retryable = False


class SessionCommitService:
    """Archive first, then independently commit memory and ordinary consumers."""

    def __init__(
        self,
        archive_store: SessionArchiveStore,
        queue_store: QueueStore,
        committer: OperationCommitter | None = None,
        memory_planner: Any | None = None,
        behavior_planner: BehaviorCommitPlanner | None = None,
        action_policy_planner: ActionPolicyCommitPlanner | None = None,
        context_planner: ContextCommitPlanner | None = None,
        session_projector: Any | None = None,
        commit_group_store: CommitGroupStore | None = None,
        memory_committer: MemoryDocumentCommitter | None = None,
        document_planner: MemoryDocumentPlanner | None = None,
        projection_journal: SessionProjectionJournal | None = None,
    ) -> None:
        self.archive_store = archive_store
        self.queue_store = queue_store
        self.committer = committer
        self.memory_planner = memory_planner
        self.behavior_planner = behavior_planner or BehaviorCommitPlanner()
        self.action_policy_planner = action_policy_planner or ActionPolicyCommitPlanner()
        self.context_planner = context_planner or ContextCommitPlanner()
        self.session_projector = session_projector
        expected_control_root = tenant_control_root(archive_store.root, archive_store.tenant_id)
        if commit_group_store is not None and commit_group_store.artifact_root != expected_control_root.resolve(
            strict=False
        ):
            raise ValueError("CommitGroupStore root differs from the bound Session tenant")
        self.commit_group_store = commit_group_store or CommitGroupStore(expected_control_root)
        self.memory_committer = memory_committer
        planner_document = getattr(memory_planner, "document_planner", None)
        if document_planner is not None and planner_document is not None and planner_document is not document_planner:
            raise ValueError("Session and memory planners must share one document planner")
        self.document_planner = document_planner or (
            planner_document if isinstance(planner_document, MemoryDocumentPlanner) else None
        )
        journal_store = getattr(session_projector, "catalog_store", None)
        self.projection_journal = projection_journal or SessionProjectionJournal(journal_store)
        self._startup_recovery_group: ContextVar[str] = ContextVar(
            f"memoryos_session_startup_recovery_{id(self)}",
            default="",
        )

    def commit_session(
        self,
        archive: SessionArchive,
        *,
        async_commit: bool = True,
    ) -> SessionCommitResult:
        return commit_session_entry(self, archive, async_commit=async_commit)

    def sync_archive(
        self,
        archive: SessionArchive,
        *,
        enqueue_commit_job: bool = True,
    ) -> SessionCommitResult:
        """Durably archive evidence before projection or queue publication."""

        self._require_runtime_ready()
        tenant_id = self._bind_archive_tenant(archive)
        tracking = False
        try:
            tracking = self._record_projection(archive, tenant_id=tenant_id, status="PENDING")
            self.archive_store.write_sync_archive(archive)
            projection, projection_status = self._project_session_archive(archive)
            if tracking:
                self._record_projection(archive, tenant_id=tenant_id, status="PROJECTED")
            if enqueue_commit_job:
                self._enqueue_session_commit(archive, tenant_id=tenant_id)
            return SessionCommitResult(
                task_id=archive.task_id,
                archive_uri=archive.archive_uri,
                status="queued",
                state=SessionCommitState.QUEUED,
                archive_committed=True,
                session_projection_status=projection_status,
                session_projected_count=int(getattr(projection, "projected", 0) or 0),
            )
        except Exception as exc:
            if tracking:
                self._record_projection(
                    archive,
                    tenant_id=tenant_id,
                    status="FAILED",
                    error=type(exc).__name__,
                )
            if self.archive_store.archive_exists(archive.archive_uri, tenant_id=tenant_id):
                self._enqueue_session_commit(archive, tenant_id=tenant_id)
            raise

    def enqueue_failed_inline_commit(self, archive: SessionArchive) -> QueueJob:
        """Publish the same durable task identity after an inline failure."""

        tenant_id = self._bind_archive_tenant(archive)
        if not self.archive_store.archive_exists(archive.archive_uri, tenant_id=tenant_id):
            raise RuntimeError("failed inline Session commit has no immutable archive")
        persisted = self.archive_store.read_archive(
            archive.archive_uri,
            tenant_id=tenant_id,
            manifest_digest=str(archive.manifest_digest or "") or None,
        )
        self._assert_archive_identity(archive, persisted)
        return self._enqueue_session_commit(persisted, tenant_id=tenant_id)

    def async_commit(self, archive: SessionArchive) -> SessionCommitResult:
        """Commit one exact archived Session through independent consumers."""

        self._require_runtime_ready()
        tenant_id = self._bind_archive_tenant(archive)
        tracking = False
        try:
            tracking = self._record_projection(archive, tenant_id=tenant_id, status="PENDING")
            requested_manifest = str(archive.manifest_digest or "")
            if not self.archive_store.archive_exists(archive.archive_uri, tenant_id=tenant_id):
                self.archive_store.write_sync_archive(archive)
                requested_manifest = archive.manifest_digest
            persisted = self.archive_store.read_archive(
                archive.archive_uri,
                tenant_id=tenant_id,
                manifest_digest=requested_manifest or None,
            )
            self._assert_archive_identity(archive, persisted, allow_materialized_digests=True)
            archive = persisted
            projection, projection_status = self._project_session_archive(archive)
            if tracking:
                self._record_projection(archive, tenant_id=tenant_id, status="PROJECTED")
        except Exception as exc:
            if tracking:
                self._record_projection(
                    archive,
                    tenant_id=tenant_id,
                    status="FAILED",
                    error=type(exc).__name__,
                )
            if self.archive_store.archive_exists(archive.archive_uri, tenant_id=tenant_id):
                self._enqueue_session_commit(archive, tenant_id=tenant_id)
            raise

        group_id = f"commit_group_{archive.task_id}"
        group = self.commit_group_store.create(
            group_id,
            task_id=archive.task_id,
            archive_uri=archive.archive_uri,
            user_id=archive.user_id,
            tenant_id=tenant_id,
            archive_digest=archive.archive_digest,
            manifest_digest=archive.manifest_digest,
        )
        if group.complete and self.archive_store.async_outputs_done_for_task(archive):
            return self._result(
                archive,
                group,
                projection_status=projection_status,
                projected_count=int(getattr(projection, "projected", 0) or 0),
            )

        failures: list[tuple[str, bool]] = []
        actions: tuple[tuple[str, Callable[[str], dict[str, Any]]], ...] = (
            ("memory", lambda attempt: self._commit_memory(archive, group_id, attempt)),
            (
                "behavior",
                self._skipped_action
                if self._is_coding_agent(archive)
                else lambda _attempt: self._commit_ordinary(
                    archive,
                    group_id,
                    "behavior",
                    self.behavior_planner.plan(archive),
                ),
            ),
            (
                "action_policy",
                self._skipped_action
                if self._is_coding_agent(archive)
                else lambda _attempt: self._commit_ordinary(
                    archive,
                    group_id,
                    "action_policy",
                    self.action_policy_planner.plan(archive),
                ),
            ),
            (
                "context",
                lambda _attempt: self._commit_ordinary(
                    archive,
                    group_id,
                    "context",
                    self.context_planner.plan(archive),
                ),
            ),
        )
        for consumer, action in actions:
            try:
                self._run_consumer(group_id, consumer, action)
            except Exception as exc:
                failures.append((consumer, self._is_retryable(exc)))

        refreshed = self.commit_group_store.load(group_id)
        if refreshed is None:
            raise KeyError(f"unknown commit group: {group_id}")
        group = refreshed
        if failures:
            self._write_outputs(archive, group, complete=False)
            raise DerivedConsumerError(failures)
        if not group.complete:
            raise DerivedConsumerError(
                tuple((name, item.retryable) for name, item in group.consumers.items() if item.status != "completed")
            )
        self._validate_persisted_memory_effects(group)
        self._write_outputs(archive, group, complete=True)
        if tracking:
            self._record_projection(archive, tenant_id=tenant_id, status="PENDING")
        try:
            projection, projection_status = self._project_session_archive(archive)
        except Exception as exc:
            if tracking:
                self._record_projection(
                    archive,
                    tenant_id=tenant_id,
                    status="FAILED",
                    error=type(exc).__name__,
                )
            raise
        if tracking:
            self._record_projection(archive, tenant_id=tenant_id, status="PROJECTED")
        return self._result(
            archive,
            group,
            projection_status=projection_status,
            projected_count=int(getattr(projection, "projected", 0) or 0),
        )

    def recover_session_projection_frontier(self, *, batch_size: int = 256) -> dict[str, int]:
        """Replay ordinary Session projection and its exact commit job."""

        if self.session_projector is None or not self.projection_journal.enabled:
            return {"projected": 0, "abandoned": 0, "failed": 0}
        self._require_runtime_ready()
        tenant_id = str(self.archive_store.tenant_id)
        maximum = max(1, min(int(batch_size), 1_000))
        counts = {"projected": 0, "abandoned": 0, "failed": 0}
        after = ""
        while True:
            entries = self.projection_journal.pending(
                tenant_id=tenant_id,
                after_archive_uri=after,
                limit=maximum,
            )
            if not entries:
                break
            for entry in entries:
                after = entry.archive_uri
                if not self.archive_store.archive_exists(entry.archive_uri, tenant_id=tenant_id):
                    self.projection_journal.mark(entry, status="ABANDONED", error="archive_missing")
                    counts["abandoned"] += 1
                    continue
                try:
                    archive = self.archive_store.read_archive(
                        entry.archive_uri,
                        tenant_id=tenant_id,
                        manifest_digest=entry.manifest_digest or None,
                    )
                    if archive.user_id != entry.owner_user_id or archive.session_id != entry.session_id:
                        raise RuntimeError("Session projection journal identity is detached")
                    self._project_session_archive(archive)
                    self._enqueue_session_commit(archive, tenant_id=tenant_id)
                    self.projection_journal.mark(entry, status="PROJECTED")
                    counts["projected"] += 1
                except Exception as exc:
                    self.projection_journal.mark(entry, status="FAILED", error=type(exc).__name__)
                    counts["failed"] += 1
                    raise RuntimeError("Session projection journal recovery failed") from exc
            if len(entries) < maximum:
                break
        return counts

    def rebuild_session_archives(
        self,
        *,
        batch_size: int = 256,
        max_archives: int = 10_000,
    ) -> dict[str, int]:
        """Rebuild derived Session Catalog rows from immutable archive heads.

        Startup calls this while the runtime is RECOVERING, so it deliberately
        bypasses the ordinary READY gate.  Enumeration and total replay are
        both bounded; reaching the configured ceiling with unseen work fails
        closed instead of publishing a partially rebuilt runtime.
        """

        if self.session_projector is None:
            return {
                "projected_archives": 0,
                "projected_records": 0,
                "async_output_archives": 0,
            }
        maximum = max(1, min(int(batch_size), 1_000))
        total_bound = int(max_archives)
        if total_bound <= 0 or total_bound > 100_000:
            raise ValueError("Session archive rebuild bound must be between 1 and 100000")
        tenant_id = str(self.archive_store.tenant_id)
        cursor = ""
        counts = {
            "projected_archives": 0,
            "projected_records": 0,
            "async_output_archives": 0,
        }
        while counts["projected_archives"] < total_bound:
            requested = min(maximum, total_bound - counts["projected_archives"])
            archives = self.archive_store.list_archives(
                tenant_id=tenant_id,
                after_archive_uri=cursor,
                limit=requested,
            )
            if not archives:
                break
            for archive in archives:
                cursor = archive.archive_uri
                try:
                    projection, _status = self._project_session_archive(
                        archive,
                        respect_applied_tombstones=True,
                    )
                    self._record_projection(
                        archive,
                        tenant_id=tenant_id,
                        status="PROJECTED",
                    )
                except Exception as exc:
                    self._record_projection(
                        archive,
                        tenant_id=tenant_id,
                        status="FAILED",
                        error=type(exc).__name__,
                    )
                    raise RuntimeError("Session archive Catalog rebuild failed") from exc
                counts["projected_archives"] += 1
                counts["projected_records"] += int(
                    getattr(projection, "projected", 0) or 0
                )
                counts["async_output_archives"] += int(
                    self.archive_store.async_outputs_done_for_task(archive)
                )
            if len(archives) < requested:
                break
        if counts["projected_archives"] >= total_bound and self.archive_store.list_archives(
            tenant_id=tenant_id,
            after_archive_uri=cursor,
            limit=1,
        ):
            raise RuntimeError("Session archive rebuild exceeded its total bound")
        return counts

    def resume_startup_commit_group(
        self,
        archive: SessionArchive,
        *,
        group_id: str,
    ) -> SessionCommitResult:
        """Validate exact archive/document effects, then replay one durable group."""

        expected_group = f"commit_group_{archive.task_id}"
        if group_id != expected_group:
            raise RuntimeError("startup archive is detached from its commit-group identity")
        tenant_id = self._bind_archive_tenant(archive)
        group = self.commit_group_store.load(group_id)
        if group is None:
            raise RuntimeError("startup commit group does not exist")
        identity = (
            archive.task_id,
            archive.archive_uri,
            archive.user_id,
            tenant_id,
            archive.archive_digest,
            archive.manifest_digest,
        )
        durable = (
            group.task_id,
            group.archive_uri,
            group.user_id,
            group.tenant_id,
            group.archive_digest,
            group.manifest_digest,
        )
        if identity != durable:
            raise RuntimeError("startup archive is detached from its durable commit group")
        self._validate_persisted_memory_effects(group)
        with self._startup_recovery_scope(group_id):
            return self.async_commit(archive)

    def resumable_commit_groups(self, *, limit: int = 256) -> tuple[CommitGroupStatus, ...]:
        """Discover unfinished groups and complete groups missing async outputs.

        A process can exit after all four consumers are durable but before the
        async-output head is published.  Such a group is terminal from the
        consumer store's perspective, so ``CommitGroupStore.pending()`` cannot
        discover it.  Recovery therefore checks the exact immutable archive
        named by each complete group before deciding that no work remains.

        This helper intentionally does not require READY: startup recovery uses
        it while the runtime is still proving its durable state.
        """

        maximum = max(1, min(int(limit), 1_000))
        actionable: list[CommitGroupStatus] = []
        leased: list[CommitGroupStatus] = []
        for group in self.commit_group_store.all():
            if not group.terminal:
                target = leased if any(item.status == "running" for item in group.consumers.values()) else actionable
                target.append(group)
            elif group.complete:
                archive = self.archive_store.read_archive_at_manifest(
                    group.archive_uri,
                    group.manifest_digest,
                    tenant_id=group.tenant_id,
                )
                identity = (
                    archive.task_id,
                    archive.archive_uri,
                    archive.user_id,
                    self._tenant_id(archive),
                    archive.archive_digest,
                    archive.manifest_digest,
                )
                durable = (
                    group.task_id,
                    group.archive_uri,
                    group.user_id,
                    group.tenant_id,
                    group.archive_digest,
                    group.manifest_digest,
                )
                if identity != durable:
                    raise RuntimeError("commit-group discovery found detached Session evidence")
                if not self.archive_store.async_outputs_done_for_task(archive):
                    actionable.append(group)
        # A long-running consumer cannot starve a later group whose consumer
        # work or output-head publication is immediately recoverable.
        return tuple((*actionable, *leased)[:maximum])

    def _commit_memory(
        self,
        archive: SessionArchive,
        group_id: str,
        attempt_id: str,
    ) -> dict[str, Any]:
        if self.memory_planner is None:
            return {
                "status": "committed",
                "edit_proposal_count": 0,
                "edit_proposal_ids": [],
                "document_change_count": 0,
                "no_op_count": 0,
            }
        plan_session = getattr(self.memory_planner, "plan_session", None)
        if not callable(plan_session):
            raise TypeError("memory planner must implement plan_session over immutable SessionArchive")
        planned = plan_session(
            archive,
            tenant_id=self._tenant_id(archive),
            owner_user_id=archive.user_id,
            commit_group_id=group_id,
        )
        edits = getattr(planned, "edits", None)
        proposal_digest = str(getattr(planned, "proposal_set_digest", "") or "")
        proposal_count = getattr(planned, "edit_proposal_count", None)
        proposal_ids = getattr(planned, "edit_proposal_ids", None)
        candidate_count = getattr(planned, "candidate_count", None)
        if not isinstance(edits, tuple):
            raise TypeError("memory planning result edits must be an immutable tuple")
        if not self._is_sha256(proposal_digest):
            raise ValueError("memory planning result must carry a sealed proposal-set digest")
        if isinstance(proposal_count, bool) or not isinstance(proposal_count, int) or proposal_count < 0:
            raise TypeError("memory planning proposal count is invalid")
        if not isinstance(proposal_ids, tuple) or any(
            not isinstance(item, str) or not item.startswith("mdreview_") for item in proposal_ids
        ):
            raise TypeError("memory planning review proposal IDs must be an immutable tuple")
        if proposal_count != len(proposal_ids):
            raise ValueError("memory planning proposal count differs from its sealed review proposals")
        if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or candidate_count < 0:
            raise TypeError("memory planning candidate count is invalid")
        if candidate_count != len(edits) + len(proposal_ids):
            raise ValueError("memory planning candidates differ from direct and review plans")
        if edits and self.memory_committer is None:
            raise RuntimeError("Markdown memory edits require MemoryDocumentCommitter")

        no_op_count = 0
        for position, edit in enumerate(edits):
            proposal = getattr(edit, "proposal", None)
            plan = getattr(edit, "plan", None)
            if not isinstance(proposal, MemoryEditProposal) or not isinstance(plan, DocumentEditPlan):
                raise TypeError("planned memory edit must bind one sealed proposal to one DocumentEditPlan")
            self._validate_document_plan(plan, archive)
            result = self._commit_document_with_replan(
                archive,
                proposal,
                plan,
                position=position,
            )
            if result.no_op:
                no_op_count += 1
                continue
            effect = self._effect_from_document_result(result)
            self.commit_group_store.record_memory_effect(
                group_id,
                effect,
                attempt_id=attempt_id,
            )

        group = self.commit_group_store.load(group_id)
        if group is None:
            raise KeyError(f"unknown commit group: {group_id}")
        self._validate_persisted_memory_effects(group)
        return {
            "status": "committed",
            "edit_proposal_count": proposal_count,
            "edit_proposal_ids": list(proposal_ids),
            "document_change_count": len(group.memory_effects),
            "no_op_count": no_op_count,
        }

    def _commit_document_with_replan(
        self,
        archive: SessionArchive,
        proposal: MemoryEditProposal,
        plan: DocumentEditPlan,
        *,
        position: int,
    ) -> DocumentCommitResult:
        assert self.memory_committer is not None
        resumed = self._resume_document_intent(archive, plan, position=position)
        if resumed is not None:
            return resumed
        try:
            return self.memory_committer.commit(
                plan,
                actor_binding=self._actor_binding(archive),
                evidence_reference=self._evidence_reference(archive, position),
            )
        except DocumentCommitConflict:
            raise
        except DocumentConflictError:
            if self.document_planner is None:
                raise
            replanned = self.document_planner.replan(
                proposal,
                tenant_id=plan.tenant_id,
                owner_user_id=plan.owner_user_id,
                idempotency_key=plan.idempotency_key,
                evidence_digest=plan.evidence_digest,
            )
            self._validate_document_plan(replanned, archive)
            resumed = self._resume_document_intent(archive, replanned, position=position)
            if resumed is not None:
                return resumed
            return self.memory_committer.commit(
                replanned,
                actor_binding=self._actor_binding(archive),
                evidence_reference=self._evidence_reference(archive, position),
            )

    def _resume_document_intent(
        self,
        archive: SessionArchive,
        plan: DocumentEditPlan,
        *,
        position: int,
    ) -> DocumentCommitResult | None:
        """Resume a source commit that crashed before its group effect was recorded."""

        assert self.memory_committer is not None
        idempotency_digest = hashlib.sha256(plan.idempotency_key.encode("utf-8")).hexdigest()
        intent_id = document_intent_id(
            plan.tenant_id,
            plan.owner_user_id,
            plan.document_id,
            idempotency_digest,
        )
        intent = self.memory_committer.control_store.load_intent(
            plan.tenant_id,
            plan.owner_user_id,
            intent_id,
        )
        if intent is None:
            return None
        if (
            intent.evidence_digest != plan.evidence_digest
            or intent.actor_binding != self._actor_binding(archive)
            or intent.evidence_reference != self._evidence_reference(archive, position)
        ):
            raise RuntimeError("document intent lineage differs from its sealed Session proposal")
        return self.memory_committer.recover_intent(
            plan.tenant_id,
            plan.owner_user_id,
            intent_id,
        )

    def _commit_ordinary(
        self,
        archive: SessionArchive,
        group_id: str,
        consumer: str,
        operations: Sequence[ContextOperation],
    ) -> dict[str, Any]:
        stabilized = self._stabilize_operations(archive, group_id, consumer, operations)
        if stabilized and self.committer is None:
            raise RuntimeError(f"{consumer} Session operations require OperationCommitter")
        diff = (
            self.committer.commit(archive.user_id, stabilized)
            if self.committer is not None
            else ContextDiff(
                user_id=archive.user_id,
                operations=[],
            )
        )
        operation_ids = [
            item.operation_id for item in (*diff.operations, *diff.pending_operations, *diff.rejected_operations)
        ]
        return {
            "status": "committed",
            "operation_count": len(operation_ids),
            "operation_ids": operation_ids,
            "diff_id": diff.diff_id,
            "skipped": False,
        }

    def _run_consumer(
        self,
        group_id: str,
        consumer: str,
        action: Callable[[str], dict[str, Any]],
    ) -> dict[str, Any]:
        if consumer not in CONSUMERS:
            raise ValueError(f"unsupported Session consumer: {consumer}")
        group = self.commit_group_store.load(group_id)
        if group is None:
            raise KeyError(f"unknown commit group: {group_id}")
        current = group.consumers[consumer]
        if current.status == "completed":
            return dict(current.summary)
        if current.status in {"dead_letter", "quarantine"} or (current.status == "failed" and not current.retryable):
            raise ConsumerTerminalError(f"{consumer} consumer is terminal")
        attempt_id = uuid.uuid4().hex
        if not self.commit_group_store.claim_consumer(
            group_id,
            consumer,
            attempt_id=attempt_id,
        ):
            raise ConsumerLeaseBusy(f"{consumer} consumer lease is unavailable")
        try:
            summary = action(attempt_id)
            self.commit_group_store.complete_consumer(
                group_id,
                consumer,
                attempt_id=attempt_id,
                summary=summary,
            )
            return summary
        except Exception as exc:
            self.commit_group_store.fail_consumer(
                group_id,
                consumer,
                type(exc).__name__,
                retryable=self._is_retryable(exc),
                attempt_id=attempt_id,
            )
            raise

    def _write_outputs(
        self,
        archive: SessionArchive,
        group: CommitGroupStatus,
        *,
        complete: bool,
    ) -> None:
        source_text = "\n".join(
            [
                *[str(item.get("content", item.get("text", ""))) for item in archive.messages],
                *[str(item.get("raw_text", item.get("scene", ""))) for item in archive.observations],
            ]
        )
        abstract = l0_abstract(source_text or f"Session {archive.session_id}")
        overview = l1_overview(
            f"Session {archive.session_id}",
            [
                f"messages: {len(archive.messages)}",
                f"observations: {len(archive.observations)}",
                f"predictions: {len(archive.predictions)}",
                f"feedback: {len(archive.feedback)}",
                "Memory documents and ordinary Session consumers are committed independently.",
            ],
        )
        memory = group.consumers["memory"]
        memory_summary = dict(memory.summary)
        memory_payload = {
            "task_id": archive.task_id,
            "commit_group_id": group.group_id,
            "status": memory_summary.get("status", memory.status),
            "edit_proposal_count": int(memory_summary.get("edit_proposal_count", 0) or 0),
            "edit_proposal_ids": list(memory_summary.get("edit_proposal_ids", []) or []),
            "memory_document_change_count": len(group.memory_effects),
            "no_op_count": int(memory_summary.get("no_op_count", 0) or 0),
            "effects": [effect.to_dict() for effect in group.memory_effects],
        }
        self.archive_store.write_async_outputs(
            archive.archive_uri,
            abstract=abstract,
            overview=overview,
            memory_diff=memory_payload,
            behavior_diff=self._ordinary_output(archive, group, "behavior"),
            action_policy_diff=self._ordinary_output(archive, group, "action_policy"),
            context_diff=self._ordinary_output(archive, group, "context"),
            tenant_id=group.tenant_id,
            commit_group_status=group.to_dict(),
            complete=complete,
            task_id=archive.task_id,
            created_at=group.created_at,
        )

    @staticmethod
    def _ordinary_output(
        archive: SessionArchive,
        group: CommitGroupStatus,
        consumer: str,
    ) -> dict[str, Any]:
        item = group.consumers[consumer]
        summary = dict(item.summary)
        return {
            "task_id": archive.task_id,
            "commit_group_id": group.group_id,
            "status": summary.get("status", item.status),
            "operation_count": int(summary.get("operation_count", 0) or 0),
            "operation_ids": list(summary.get("operation_ids", []) or []),
            "diff_id": str(summary.get("diff_id", "") or ""),
            "skipped": bool(summary.get("skipped", False)),
        }

    def _result(
        self,
        archive: SessionArchive,
        group: CommitGroupStatus,
        *,
        projection_status: str = "not_configured",
        projected_count: int = 0,
    ) -> SessionCommitResult:
        if group.complete:
            status = "done"
            state = SessionCommitState.COMMITTED
            done = True
        elif group.terminal:
            status = "dead_letter"
            state = SessionCommitState.DEAD_LETTER
            done = False
        else:
            status = "retrying"
            state = SessionCommitState.FAILED_RETRYABLE
            done = False
        memory_summary = group.consumers["memory"].summary
        return SessionCommitResult(
            task_id=archive.task_id,
            archive_uri=archive.archive_uri,
            status=status,
            done=done,
            state=state,
            commit_group_id=group.group_id,
            memory_committed=group.memory_committed,
            commit_group_status=group.to_dict(),
            archive_committed=True,
            memory_document_change_count=len(group.memory_effects),
            edit_proposal_count=int(memory_summary.get("edit_proposal_count", 0) or 0),
            edit_proposal_ids=tuple(memory_summary.get("edit_proposal_ids", []) or ()),
            session_projection_status=projection_status,
            session_projected_count=projected_count,
        )

    def _validate_persisted_memory_effects(self, group: CommitGroupStatus) -> None:
        if not group.memory_effects:
            return
        if self.memory_committer is None:
            raise RuntimeError("durable memory effects require MemoryDocumentCommitter")
        for effect in group.memory_effects:
            binding = self.memory_committer.control_store.load_event_binding(
                group.tenant_id,
                group.user_id,
                effect.document_id,
                effect.change_event_id,
            )
            if binding is None:
                barrier = self.memory_committer.control_store.load_publication_barrier(
                    group.tenant_id,
                    group.user_id,
                    effect.document_id,
                )
                if barrier is not None and barrier.status is DocumentDeletionStatus.HARD_ERASED:
                    continue
                raise RuntimeError("commit-group memory effect is detached from its change event")
            intent, event = binding
            if intent.status is not DocumentIntentStatus.COMPLETED:
                raise RuntimeError("commit-group memory effect is detached from its completed intent")
            if canonical_digest(event.to_dict()) != effect.change_digest:
                raise RuntimeError("commit-group memory effect is detached from its change event")
            job = self.memory_committer.projection_queue.get(intent.projection_job_id)
            if (
                job is None
                or job.queue_name != "memory_projection"
                or job.action != "memory_committed"
                or job.payload.get("intent_id") != intent.intent_id
                or job.payload.get("document_id") != intent.document_id
                or job.payload.get("event_id") != intent.event_id
            ):
                raise RuntimeError("completed memory intent has no durable projection job")

    def _effect_from_document_result(self, result: DocumentCommitResult) -> MemoryDocumentEffect:
        if result.status is not DocumentIntentStatus.COMPLETED or result.event is None:
            raise RuntimeError("document committer did not complete its source and projection enqueue")
        if result.event.document_id == "":
            raise RuntimeError("document change event has no document identity")
        return MemoryDocumentEffect(
            document_id=result.event.document_id,
            change_event_id=result.event.event_id,
            change_digest=canonical_digest(result.event.to_dict()),
        )

    def _validate_document_plan(self, plan: DocumentEditPlan, archive: SessionArchive) -> None:
        if plan.tenant_id != self._tenant_id(archive) or plan.owner_user_id != archive.user_id:
            raise PermissionError("document plan crosses the archived Session boundary")
        if not self._is_sha256(plan.evidence_digest):
            raise ValueError("document plan evidence digest is invalid")
        if not plan.idempotency_key:
            raise ValueError("document plan idempotency key is empty")

    def _stabilize_operations(
        self,
        archive: SessionArchive,
        group_id: str,
        consumer: str,
        operations: Sequence[ContextOperation],
    ) -> list[ContextOperation]:
        stable: list[ContextOperation] = []
        for position, operation in enumerate(operations):
            if not isinstance(operation, ContextOperation):
                raise TypeError(f"{consumer} planner returned a non-ContextOperation")
            if operation.user_id != archive.user_id:
                raise PermissionError(f"{consumer} operation crosses the Session owner boundary")
            operation_key = stable_hash(
                [
                    group_id,
                    consumer,
                    position,
                    operation.context_type.value,
                    operation.action.value,
                    operation.target_uri,
                ],
                40,
            )
            operation.operation_id = f"op_{operation_key}"
            operation.created_at = archive.created_at
            operation.payload["commit_group_id"] = group_id
            operation.payload["commit_consumer"] = consumer
            stable.append(operation)
        return stable

    def _enqueue_session_commit(self, archive: SessionArchive, *, tenant_id: str) -> QueueJob:
        return self.queue_store.enqueue(
            QueueJob(
                job_id=archive.task_id,
                queue_name="session_commit",
                action="async_session_commit",
                target_uri=archive.archive_uri,
                payload={
                    "user_id": archive.user_id,
                    "session_id": archive.session_id,
                    "tenant_id": tenant_id,
                    "archive_digest": archive.archive_digest,
                    "manifest_digest": archive.manifest_digest,
                },
            )
        )

    def _project_session_archive(
        self,
        archive: SessionArchive,
        *,
        respect_applied_tombstones: bool = False,
    ) -> tuple[Any | None, str]:
        if self.session_projector is None:
            return None, "not_configured"
        async_outputs: dict[str, Any] | None = None
        if self.archive_store.async_outputs_done_for_task(archive):
            async_outputs = self.archive_store.read_async_outputs(archive)
        elif str(getattr(self.archive_store, "last_async_output_error", "") or ""):
            raise RuntimeError("Session async outputs failed integrity validation")
        kwargs: dict[str, Any] = {}
        if async_outputs is not None:
            kwargs["async_outputs"] = async_outputs
        if respect_applied_tombstones:
            kwargs["respect_applied_tombstones"] = True
        return self.session_projector.project(archive, **kwargs), "projected"

    def _record_projection(
        self,
        archive: SessionArchive,
        *,
        tenant_id: str,
        status: str,
        error: str = "",
    ) -> bool:
        if self.session_projector is None or not self.projection_journal.enabled:
            return False
        return self.projection_journal.record(
            archive,
            tenant_id=tenant_id,
            status=status,
            error=error,
        )

    def _require_runtime_ready(self) -> None:
        committer = getattr(self.committer, "delegate", self.committer)
        source_store = getattr(committer, "source_store", None)
        if source_store is None:
            source_store = getattr(self.memory_planner, "source_store", None)
        readiness = getattr(source_store, "readiness", None)
        require_ready = getattr(readiness, "require_ready", None)
        if not callable(require_ready) or self._startup_recovery_group.get():
            return
        require_ready()

    @contextmanager
    def _startup_recovery_scope(self, group_id: str) -> Iterator[None]:
        token = self._startup_recovery_group.set(group_id)
        committer = getattr(self.committer, "delegate", self.committer)
        scope = getattr(committer, "_durable_startup_recovery_scope", None)
        try:
            if callable(scope):
                with cast(AbstractContextManager[None], scope(group_id)):
                    yield
            else:
                yield
        finally:
            self._startup_recovery_group.reset(token)

    def _tenant_id(self, archive: SessionArchive) -> str:
        metadata = dict(archive.metadata or {})
        scope = dict(metadata.get("scope", {}) or {})
        bound = str(self.archive_store.tenant_id)
        claimed = tuple(
            str(value) for value in (metadata.get("tenant_id"), scope.get("tenant_id")) if value not in (None, "")
        )
        if any(value != bound for value in claimed):
            raise PermissionError("SessionArchive tenant differs from its bound store")
        return bound

    def _bind_archive_tenant(self, archive: SessionArchive) -> str:
        tenant_id = self._tenant_id(archive)
        metadata = dict(archive.metadata or {})
        metadata["tenant_id"] = tenant_id
        if "scope" in metadata:
            scope = dict(metadata.get("scope", {}) or {})
            scope["tenant_id"] = tenant_id
            metadata["scope"] = scope
        archive.metadata = metadata
        return tenant_id

    @staticmethod
    def _assert_archive_identity(
        requested: SessionArchive,
        persisted: SessionArchive,
        *,
        allow_materialized_digests: bool = False,
    ) -> None:
        if (
            requested.task_id != persisted.task_id
            or requested.user_id != persisted.user_id
            or requested.session_id != persisted.session_id
            or requested.archive_uri != persisted.archive_uri
        ):
            raise RuntimeError("SessionArchive request is detached from durable evidence")
        for field in ("archive_digest", "manifest_digest"):
            expected = str(getattr(requested, field, "") or "")
            actual = str(getattr(persisted, field, "") or "")
            if expected and expected != actual:
                raise RuntimeError(f"SessionArchive {field} changed across durable read")
            if not expected and not allow_materialized_digests:
                raise RuntimeError(f"SessionArchive {field} is not materialized")

    @staticmethod
    def _actor_binding(archive: SessionArchive) -> str:
        return f"session:{archive.task_id}:{archive.user_id}"

    @staticmethod
    def _evidence_reference(archive: SessionArchive, position: int) -> str:
        manifest = archive.manifest_uri or f"{archive.archive_uri}#manifest={archive.manifest_digest}"
        return f"{manifest}:memory-edit:{position}"

    @staticmethod
    def _skipped_action(_attempt_id: str) -> dict[str, Any]:
        return {
            "status": "skipped",
            "operation_count": 0,
            "operation_ids": [],
            "diff_id": "",
            "skipped": True,
        }

    @staticmethod
    def _is_coding_agent(archive: SessionArchive) -> bool:
        connect = dict(archive.metadata.get("connect", {}) or {})
        return connect.get("connect_type") == "agent" and connect.get("run_mode") == "context_reduction"

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        explicit = getattr(exc, "retryable", None)
        if isinstance(explicit, bool):
            return explicit
        return isinstance(exc, (OSError, TimeoutError, RevisionConflictError, DocumentConflictError))

    @staticmethod
    def _is_sha256(value: str) -> bool:
        return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


__all__ = [
    "ConsumerLeaseBusy",
    "ConsumerTerminalError",
    "DerivedConsumerError",
    "SessionCommitService",
]
