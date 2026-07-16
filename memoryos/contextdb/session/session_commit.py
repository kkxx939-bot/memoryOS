"""上下文数据库里的会话提交。"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any, cast

from memoryos.contextdb.layers.layer_generator import l0_abstract, l1_overview
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.session.commit_group import CommitGroupStatus, CommitGroupStore
from memoryos.contextdb.session.context_projector import workspace_id_from_session_metadata
from memoryos.contextdb.session.planners import (
    ActionPolicyCommitPlanner,
    BehaviorCommitPlanner,
    ContextCommitPlanner,
    MemoryCommitPlanner,
)
from memoryos.contextdb.session.planners.memory_commit_planner import MemoryExtractionBackendError
from memoryos.contextdb.session.planning import MemoryPlanningResult, PlanningContext
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.session.session_model import SessionArchive, SessionCommitResult, SessionCommitState
from memoryos.contextdb.store.source_store import QueueJob, QueueStore
from memoryos.core.ids import stable_hash
from memoryos.memory.canonical.transaction import RevisionConflictError
from memoryos.memory.canonical.visibility import read_committed_pending
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation


class DerivedConsumerError(RuntimeError):
    def __init__(
        self,
        consumer: str,
        failures: list[str],
        *,
        retryable: bool = True,
    ) -> None:
        self.consumer = consumer
        self.failures = tuple(failures)
        self.retryable = retryable
        super().__init__(f"{consumer} consumer failed: {','.join(failures)}")


class SessionCommitService:
    """先归档会话，再规划并提交各类上下文变更。"""

    def __init__(
        self,
        archive_store: SessionArchiveStore,
        queue_store: QueueStore,
        committer: OperationCommitter | None = None,
        memory_planner: MemoryCommitPlanner | None = None,
        behavior_planner: BehaviorCommitPlanner | None = None,
        action_policy_planner: ActionPolicyCommitPlanner | None = None,
        context_planner: ContextCommitPlanner | None = None,
        allow_plan_only: bool = False,
        projection_worker=None,
        session_projector=None,
        migration_gate=None,
        commit_group_store: CommitGroupStore | None = None,
    ) -> None:
        self.archive_store = archive_store
        self.queue_store = queue_store
        self.committer = committer
        self.memory_planner = memory_planner or MemoryCommitPlanner()
        bind_runtime_stores = getattr(self.memory_planner, "bind_runtime_stores", None)
        if committer is not None and callable(bind_runtime_stores):
            binding_committer = getattr(committer, "delegate", committer)
            required = ("source_store", "index_store", "relation_store", "root", "tenant_id")
            if all(hasattr(binding_committer, name) for name in required):
                bind_runtime_stores(
                    binding_committer.source_store,
                    binding_committer.index_store,
                    binding_committer.relation_store,
                    root=binding_committer.root,
                    tenant_id=binding_committer.tenant_id,
                )
        self.behavior_planner = behavior_planner or BehaviorCommitPlanner()
        self.action_policy_planner = action_policy_planner or ActionPolicyCommitPlanner()
        self.context_planner = context_planner or ContextCommitPlanner()
        self.allow_plan_only = allow_plan_only
        self.projection_worker = projection_worker
        self.session_projector = session_projector
        self.migration_gate = migration_gate
        self.commit_group_store = commit_group_store or CommitGroupStore(archive_store.root)

    def _project_session_archive(self, archive: SessionArchive) -> tuple[Any | None, str]:
        """Apply the durable migration dual-write gate to Session serving rows."""

        if self.session_projector is None:
            return None, "not_configured"
        if self.migration_gate is not None:
            feature_gate = getattr(self.migration_gate, "feature_gate", None)
            if feature_gate is None or not bool(getattr(feature_gate, "dual_write_enabled", False)):
                return None, "migration_legacy_only"
        result = self.session_projector.project(archive)
        recorder = getattr(self.migration_gate, "record_projection_equivalence", None)
        proof = getattr(result, "equivalence_proof", None)
        if callable(recorder):
            if proof is None:
                state = str(
                    getattr(
                        getattr(getattr(self.migration_gate, "feature_gate", None), "state", None),
                        "value",
                        "",
                    )
                )
                if state == "SHADOW_VALIDATING":
                    raise RuntimeError("shadow Session projection has no independent equivalence proof")
            else:
                recorder(proof)
        return result, "projected"

    def _require_runtime_ready(self) -> None:
        committer = getattr(self.committer, "delegate", self.committer)
        source_store = getattr(committer, "source_store", None)
        if source_store is None:
            source_store = getattr(self.memory_planner, "source_store", None)
        readiness = getattr(source_store, "readiness", None)
        require_ready = getattr(readiness, "require_ready", None)
        if not callable(require_ready):
            return
        state = str(getattr(getattr(readiness, "state", None), "value", ""))
        recovery_group = getattr(committer, "_startup_recovery_group", None)
        recovery_group_get = getattr(recovery_group, "get", None)
        if state == "RECOVERING" and callable(recovery_group_get) and recovery_group_get():
            return
        require_ready()

    def sync_archive(self, archive: SessionArchive, *, enqueue_commit_job: bool = True) -> SessionCommitResult:
        """先把原始会话证据写稳，再投递异步提交任务。"""

        self._require_runtime_ready()
        fence = self._acquire_migration_projection_fence()
        tracking = False
        try:
            tenant_id = self._bind_archive_tenant(archive)
            tracking = self._record_session_projection_frontier(archive, status="PENDING")
            self.archive_store.write_sync_archive(archive)
            if enqueue_commit_job:
                self._enqueue_session_commit(archive, tenant_id=tenant_id)
            # The immutable evidence and (on failure) durable replay job exist
            # before releasing the migration cutover fence.
            session_projection, session_projection_status = self._project_session_archive(archive)
            if tracking:
                self._record_session_projection_frontier(archive, status="PROJECTED")
            return SessionCommitResult(
                task_id=archive.task_id,
                archive_uri=archive.archive_uri,
                status="queued",
                state=SessionCommitState.QUEUED,
                archive_committed=True,
                session_projection_status=session_projection_status,
                session_projected_count=int(getattr(session_projection, "projected", 0) or 0),
            )
        except Exception as exc:
            if tracking:
                self._record_session_projection_frontier(
                    archive,
                    status="FAILED",
                    error=f"{type(exc).__name__}: {exc}",
                )
            tenant_id = self._bind_archive_tenant(archive)
            if self.archive_store.archive_exists(archive.archive_uri, tenant_id=tenant_id):
                self._enqueue_session_commit(archive, tenant_id=tenant_id)
            raise
        finally:
            self._release_migration_projection_fence(fence)

    def enqueue_failed_inline_commit(self, archive: SessionArchive) -> QueueJob:
        """Durably recover an inline commit after its immutable archive exists.

        Successful ``async_commit=True`` calls retain their historical no-job
        behavior.  This helper is invoked only after an inline projection or
        consumer raises, closing the BACKFILLING checkpoint race without
        inventing evidence for a session that was never archived.
        """

        fence = self._acquire_migration_projection_fence()
        try:
            tenant_id = self._bind_archive_tenant(archive)
            if not self.archive_store.archive_exists(archive.archive_uri, tenant_id=tenant_id):
                raise RuntimeError("failed inline Session commit has no durable archive evidence")
            return self._enqueue_session_commit(archive, tenant_id=tenant_id)
        finally:
            self._release_migration_projection_fence(fence)

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
                    "archive_digest": str(getattr(archive, "archive_digest", "") or ""),
                    "manifest_digest": str(getattr(archive, "manifest_digest", "") or ""),
                },
            )
        )

    def async_commit(self, archive: SessionArchive) -> SessionCommitResult:
        """根据已归档会话生成并提交记忆、行为和上下文变更。"""

        self._require_runtime_ready()
        fence = self._acquire_migration_projection_fence()
        tracking = False
        try:
            tenant_id = self._bind_archive_tenant(archive)
            tracking = self._record_session_projection_frontier(archive, status="PENDING")
            requested_manifest = str(archive.manifest_digest or "")
            if not self.archive_store.archive_exists(archive.archive_uri, tenant_id=tenant_id):
                self.archive_store.write_sync_archive(archive)
                requested_manifest = archive.manifest_digest
            archive = self.archive_store.read_archive(
                archive.archive_uri,
                tenant_id=tenant_id,
                manifest_digest=requested_manifest or None,
            )
            self._project_session_archive(archive)
            if tracking:
                self._record_session_projection_frontier(archive, status="PROJECTED")
        except Exception as exc:
            if tracking:
                self._record_session_projection_frontier(
                    archive,
                    status="FAILED",
                    error=f"{type(exc).__name__}: {exc}",
                )
            tenant_id = self._bind_archive_tenant(archive)
            if self.archive_store.archive_exists(archive.archive_uri, tenant_id=tenant_id):
                self._enqueue_session_commit(archive, tenant_id=tenant_id)
            raise
        finally:
            self._release_migration_projection_fence(fence)
        tenant_id = self._bind_archive_tenant(archive)
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
        if self.committer is None and not self.allow_plan_only:
            raise RuntimeError("SessionCommitService requires OperationCommitter unless allow_plan_only=True")
        if group.complete and self.archive_store.async_outputs_done_for_task(archive):
            return self._result(archive, group)
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
                "Long-term memory, behavior, action policy, and context diffs are emitted separately.",
            ],
        )
        if group.canonical_status != "completed":
            if group.canonical_status in {"dead_letter", "quarantine"} or (
                group.canonical_status == "failed" and not group.canonical_retryable
            ):
                return self._write_incomplete_outputs(
                    archive,
                    group,
                    abstract,
                    overview,
                    memory_diff=self._partial_memory_result(group),
                )
            canonical_attempt_id = uuid.uuid4().hex
            if not self.commit_group_store.claim_canonical(
                group_id,
                attempt_id=canonical_attempt_id,
            ):
                current = self.commit_group_store.load(group_id)
                assert current is not None
                return self._result(archive, current)
            try:
                if self.committer is not None:
                    self._recover_pending_memory_effects(archive.user_id, group_id)
                    self._backfill_canonical_effects(group_id, archive.user_id)
                plan_with_progress = getattr(self.memory_planner, "plan_with_progress", None)
                if callable(plan_with_progress):
                    memory_result = cast(
                        MemoryPlanningResult,
                        plan_with_progress(
                            archive,
                            progress=lambda phase, digest: self.commit_group_store.mark_canonical_phase(
                                group_id,
                                phase=phase,
                                attempt_id=canonical_attempt_id,
                                salience_reservation_digest=digest,
                            ),
                        ),
                    )
                else:
                    memory_result = self.memory_planner.plan(archive)
                if (
                    len(memory_result.context.salience_reservation_digest) == 64
                    and len(memory_result.context.planning_digest) == 64
                ):
                    self.commit_group_store.mark_canonical_phase(
                        group_id,
                        phase="planning_sealed",
                        attempt_id=canonical_attempt_id,
                        salience_reservation_digest=memory_result.context.salience_reservation_digest,
                        planning_digest=memory_result.context.planning_digest,
                    )
                memory_ops = list(memory_result.operations)
                memory_diff = self._commit_memory_with_reconcile_retry(
                    archive,
                    memory_ops,
                    memory_result.context,
                    commit_group_id=group_id,
                )
            except MemoryExtractionBackendError as exc:
                if exc.retryable:
                    self._enqueue_memory_proposal(archive, exc)
                group = self.commit_group_store.fail_canonical(
                    group_id,
                    exc.error_type,
                    retryable=exc.retryable,
                    attempt_id=canonical_attempt_id,
                )
                return self._write_incomplete_outputs(
                    archive,
                    group,
                    abstract,
                    overview,
                    memory_diff={
                        "status": "pending",
                        "proposal_status": "queued" if exc.retryable else "dead_letter",
                        "error": exc.error_type,
                        "operation_count": 0,
                        "operations": [],
                    },
                )
            except (OSError, TimeoutError, RevisionConflictError) as exc:
                self.commit_group_store.fail_canonical(
                    group_id,
                    type(exc).__name__,
                    retryable=True,
                    attempt_id=canonical_attempt_id,
                )
                raise
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                self.commit_group_store.fail_canonical(
                    group_id,
                    type(exc).__name__,
                    retryable=False,
                    attempt_id=canonical_attempt_id,
                )
                raise
            except Exception as exc:
                # Unknown internal failures are configuration/code failures,
                # not transport failures. Release the canonical lease before
                # propagating so the group cannot remain indefinitely running.
                self.commit_group_store.fail_canonical(
                    group_id,
                    type(exc).__name__,
                    retryable=False,
                    attempt_id=canonical_attempt_id,
                )
                raise
            group = self.commit_group_store.mark_canonical(
                group_id,
                revision=self._max_revision_from_diff(memory_diff),
                result=memory_diff,
                attempt_id=canonical_attempt_id,
            )
        memory_diff = dict(group.canonical_result)
        coding_agent = self._is_coding_agent(archive)
        projection_diff = self._run_consumer(
            group_id,
            "projection",
            lambda: self._project_commit_group(group_id, memory_diff),
        )
        memory_diff["projection"] = projection_diff
        if coding_agent:
            behavior_diff = self._complete_skipped_consumer(group_id, "behavior")
        else:
            behavior_diff = self._run_consumer(
                group_id,
                "behavior",
                lambda: self._commit_consumer_operations(
                    archive,
                    group_id,
                    "behavior",
                    self.behavior_planner.plan(archive),
                ),
            )
        if coding_agent:
            action_policy_diff = self._complete_skipped_consumer(group_id, "action_policy")
        else:
            action_policy_diff = self._run_consumer(
                group_id,
                "action_policy",
                lambda: self._commit_consumer_operations(
                    archive,
                    group_id,
                    "action_policy",
                    self.action_policy_planner.plan(archive),
                ),
            )
        context_diff = self._run_consumer(
            group_id,
            "context",
            lambda: self._commit_consumer_operations(
                archive,
                group_id,
                "context",
                self.context_planner.plan(archive),
            ),
        )
        refreshed_group = self.commit_group_store.load(group_id)
        assert refreshed_group is not None
        group = refreshed_group
        self.archive_store.write_async_outputs(
            archive.archive_uri,
            abstract=abstract,
            overview=overview,
            memory_diff={"task_id": archive.task_id, "commit_group_id": group_id, **memory_diff},
            behavior_diff={"task_id": archive.task_id, **behavior_diff},
            action_policy_diff={"task_id": archive.task_id, **action_policy_diff},
            context_diff={"task_id": archive.task_id, **context_diff},
            tenant_id=tenant_id,
            commit_group_status=group.to_dict(),
            complete=group.complete,
        )
        return self._result(archive, group)

    def _acquire_migration_projection_fence(self) -> Any | None:
        acquire = getattr(self.migration_gate, "acquire_projection_fence", None)
        return acquire() if callable(acquire) else None

    def _release_migration_projection_fence(self, token: Any | None) -> None:
        release = getattr(self.migration_gate, "release_projection_fence", None)
        if callable(release):
            release(token)

    def _record_session_projection_frontier(
        self,
        archive: SessionArchive,
        *,
        status: str,
        error: str = "",
    ) -> bool:
        if self.session_projector is None:
            return False
        feature_gate = getattr(self.migration_gate, "feature_gate", None)
        if feature_gate is not None and not bool(getattr(feature_gate, "dual_write_enabled", False)):
            return False
        store = getattr(self.session_projector, "catalog_store", None)
        recorder = getattr(store, "set_session_projection_frontier", None)
        if not callable(recorder):
            return False
        recorder(
            tenant_id=self._bind_archive_tenant(archive),
            archive_uri=archive.archive_uri,
            owner_user_id=archive.user_id,
            workspace_id=workspace_id_from_session_metadata(archive.metadata),
            session_id=archive.session_id,
            manifest_digest=str(archive.manifest_digest or ""),
            status=status,
            error=error,
        )
        return True

    def recover_session_projection_frontier(self, *, batch_size: int = 256) -> dict[str, int]:
        fence = self._acquire_migration_projection_fence()
        try:
            return self._recover_session_projection_frontier_unfenced(batch_size=batch_size)
        finally:
            self._release_migration_projection_fence(fence)

    def _recover_session_projection_frontier_unfenced(self, *, batch_size: int = 256) -> dict[str, int]:
        """Repair crash windows between Archive publish, queueing, and projection.

        The scan is tenant-keyset paginated.  Every row is rebound to an
        immutable Archive manifest before an idempotent queue job or Catalog
        projection is created; a pre-write frontier without Evidence becomes
        an explicit terminal ABANDONED record instead of poisoning cutover.
        """

        if self.session_projector is None:
            return {"scanned": 0, "projected": 0, "enqueued": 0, "abandoned": 0}
        store = cast(Any, getattr(self.session_projector, "catalog_store", None))
        if store is None:
            return {"scanned": 0, "projected": 0, "enqueued": 0, "abandoned": 0}
        lister = getattr(store, "list_session_projection_frontier", None)
        if not callable(lister):
            return {"scanned": 0, "projected": 0, "enqueued": 0, "abandoned": 0}
        tenant_id = str(self.archive_store.tenant_id)
        maximum = max(1, min(int(batch_size), 1_000))
        counts = {"scanned": 0, "projected": 0, "enqueued": 0, "abandoned": 0}
        after = ""
        while True:
            rows = cast(
                list[dict[str, Any]],
                lister(
                    tenant_id=tenant_id,
                    statuses=("PENDING", "FAILED"),
                    after_archive_uri=after,
                    limit=maximum,
                ),
            )
            if not rows:
                break
            for row in rows:
                archive: SessionArchive | None = None
                after = str(row.get("archive_uri") or "")
                counts["scanned"] += 1
                if not self.archive_store.archive_exists(after, tenant_id=tenant_id):
                    store.set_session_projection_frontier(
                        tenant_id=tenant_id,
                        archive_uri=after,
                        owner_user_id=str(row.get("owner_user_id") or ""),
                        workspace_id=str(row.get("workspace_id") or ""),
                        session_id=str(row.get("session_id") or ""),
                        manifest_digest=str(row.get("manifest_digest") or ""),
                        status="ABANDONED",
                        error="immutable SessionArchive evidence was not published",
                    )
                    counts["abandoned"] += 1
                    continue
                try:
                    archive = self.archive_store.read_archive(
                        after,
                        tenant_id=tenant_id,
                        manifest_digest=str(row.get("manifest_digest") or "") or None,
                    )
                    if archive.session_id != str(row.get("session_id") or ""):
                        raise ValueError("Session projection frontier session identity mismatch")
                    self._enqueue_session_commit(archive, tenant_id=tenant_id)
                    counts["enqueued"] += 1
                    projection, status = self._project_session_archive(archive)
                    if status != "projected" or projection is None:
                        raise RuntimeError("Session projection recovery is outside the dual-write gate")
                    self._record_session_projection_frontier(archive, status="PROJECTED")
                    counts["projected"] += 1
                except Exception as exc:
                    store.set_session_projection_frontier(
                        tenant_id=tenant_id,
                        archive_uri=after,
                        owner_user_id=str(
                            row.get("owner_user_id") or (archive.user_id if archive else "")
                        ),
                        workspace_id=str(
                            row.get("workspace_id")
                            or (
                                workspace_id_from_session_metadata(archive.metadata)
                                if archive
                                else ""
                            )
                        ),
                        session_id=str(row.get("session_id") or ""),
                        manifest_digest=str(row.get("manifest_digest") or ""),
                        status="FAILED",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    raise RuntimeError("Session projection frontier recovery failed") from exc
            if len(rows) < maximum:
                break
        return counts

    def resume_startup_commit_group(
        self,
        archive: SessionArchive,
        *,
        group_id: str,
    ) -> SessionCommitResult:
        """Resume one durable group without opening a RECOVERING write bypass."""

        if self.committer is None:
            raise RuntimeError("startup commit-group recovery requires OperationCommitter")
        expected_group_id = f"commit_group_{archive.task_id}"
        if group_id != expected_group_id:
            raise RuntimeError("startup archive is detached from its commit-group identity")
        group = self.commit_group_store.load(group_id)
        tenant_id = self._bind_archive_tenant(archive)
        if (
            group is None
            or group.task_id != archive.task_id
            or group.archive_uri != archive.archive_uri
            or group.user_id != archive.user_id
            or group.tenant_id != tenant_id
            or group.archive_digest != archive.archive_digest
            or group.manifest_digest != archive.manifest_digest
            or group.complete
        ):
            raise RuntimeError("startup archive is detached from its durable commit group")
        if group.canonical_status != "completed":
            envelope = self.committer.planning_envelopes.load_validated_payload(group.task_id)
            if (
                envelope.get("operation_group_identity") != group.group_id
                or envelope.get("archive_uri") != group.archive_uri
                or envelope.get("archive_digest") != group.archive_digest
                or envelope.get("manifest_digest") != group.manifest_digest
                or envelope.get("user_id") != group.user_id
                or envelope.get("tenant_id") != group.tenant_id
                or (group.planning_digest and envelope.get("planning_digest") != group.planning_digest)
            ):
                raise RuntimeError("startup commit group is detached from durable planning")
        with self.committer._durable_startup_recovery_scope(group_id):
            return self.async_commit(archive)

    def _commit_memory_with_reconcile_retry(
        self,
        archive: SessionArchive,
        operations: list[ContextOperation],
        planning_context: PlanningContext | None = None,
        *,
        commit_group_id: str = "",
    ) -> dict:
        planned_operations = list(operations)
        try:
            try:
                committed = self._commit_or_describe(archive.user_id, operations)
            except RevisionConflictError as exc:
                partial_diff = exc.committed_diff
                if self.committer is not None:
                    self._recover_pending_memory_effects(
                        archive.user_id,
                        commit_group_id,
                    )
                if planning_context is None:
                    raise RevisionConflictError("revision conflict has no request-scoped PlanningContext") from exc
                replanned = self.memory_planner.replan(planning_context, archive)
                planned_operations.extend(replanned.operations)
                if self.committer is not None:
                    retried_diff = self.committer.commit(archive.user_id, list(replanned.operations))
                    combined_diff = self.committer.combine_committed_diffs(
                        archive.user_id,
                        [item for item in (partial_diff, retried_diff) if item is not None],
                    )
                    committed = self._diff_payload(combined_diff, status="committed")
                else:
                    retried = self._commit_or_describe(archive.user_id, list(replanned.operations))
                    partial = self._diff_payload(partial_diff, status="committed") if partial_diff is not None else None
                    committed = self._merge_diff_payloads(*(item for item in (partial, retried) if item is not None))
        except BaseException as original_error:
            if commit_group_id and self.committer is not None:
                recovery_error: Exception | None = None
                try:
                    self._recover_pending_memory_effects(archive.user_id, commit_group_id)
                except (OSError, TimeoutError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    recovery_error = exc
                try:
                    self._backfill_canonical_effects(commit_group_id, archive.user_id)
                except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    if recovery_error is None:
                        recovery_error = exc
                if recovery_error is not None:
                    raise recovery_error from original_error
            raise
        if commit_group_id and self.committer is not None:
            self._backfill_canonical_effects(commit_group_id, archive.user_id)
            committed = self._merge_canonical_effects(
                commit_group_id,
                archive.user_id,
                committed,
            )
        return self._memory_commit_metadata(committed, planned_operations)

    def _run_consumer(
        self,
        group_id: str,
        consumer: str,
        action: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        status = self.commit_group_store.load(group_id)
        if status is None:
            raise KeyError(f"unknown commit group: {group_id}")
        existing = status.consumers[consumer]
        if existing.status == "completed":
            return dict(existing.result) or {"status": "completed"}
        if existing.status in {"dead_letter", "quarantine"} or (existing.status == "failed" and not existing.retryable):
            return {"status": existing.status, "error": existing.last_error, "retryable": False}
        attempt_id = uuid.uuid4().hex
        if not self.commit_group_store.claim_consumer(
            group_id,
            consumer,
            attempt_id=attempt_id,
        ):
            current = self.commit_group_store.load(group_id)
            assert current is not None
            item = current.consumers[consumer]
            return {
                "status": item.status,
                "error": item.last_error,
                "retryable": item.retryable,
            }
        try:
            result = action()
            failures = [str(item) for item in result.get("failed", []) or []]
            if failures:
                raise DerivedConsumerError(
                    consumer,
                    failures,
                    retryable=bool(result.get("retryable", True)),
                )
        except DerivedConsumerError as exc:
            failed = self.commit_group_store.fail_consumer(
                group_id,
                consumer,
                type(exc).__name__,
                retryable=exc.retryable,
                attempt_id=attempt_id,
            )
            item = failed.consumers[consumer]
            return {
                "status": item.status,
                "error": item.last_error,
                "retryable": item.retryable,
            }
        except (OSError, TimeoutError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            failed = self.commit_group_store.fail_consumer(
                group_id,
                consumer,
                type(exc).__name__,
                retryable=True,
                attempt_id=attempt_id,
            )
            item = failed.consumers[consumer]
            return {
                "status": item.status,
                "error": item.last_error,
                "retryable": item.retryable,
            }
        except Exception as exc:
            failed = self.commit_group_store.fail_consumer(
                group_id,
                consumer,
                type(exc).__name__,
                retryable=False,
                attempt_id=attempt_id,
            )
            item = failed.consumers[consumer]
            return {
                "status": item.status,
                "error": item.last_error,
                "retryable": item.retryable,
            }
        current = self.commit_group_store.load(group_id)
        assert current is not None
        self.commit_group_store.complete_consumer(
            group_id,
            consumer,
            revision=current.canonical_revision,
            attempt_id=attempt_id,
            result=result,
        )
        return result

    def _complete_skipped_consumer(self, group_id: str, consumer: str) -> dict[str, Any]:
        result = {"status": "skipped", "operations": [], "operation_count": 0}
        status = self.commit_group_store.load(group_id)
        if status is None:
            raise KeyError(f"unknown commit group: {group_id}")
        if status.consumers[consumer].status != "completed":
            self.commit_group_store.complete_consumer(
                group_id,
                consumer,
                revision=status.canonical_revision,
                result=result,
            )
        return result

    def _project_commit_group(self, group_id: str, memory_diff: dict[str, Any]) -> dict[str, Any]:
        transaction_ids = tuple(
            dict.fromkeys(
                str(operation.get("payload", {}).get("transaction_id", ""))
                for operation in memory_diff.get("operations", []) or []
                if isinstance(operation, dict)
                and operation.get("payload", {}).get("canonical_memory") is True
                and operation.get("payload", {}).get("transaction_id")
            )
        )
        if self.projection_worker is None:
            if transaction_ids:
                return {
                    "status": "failed",
                    "processed": [],
                    "stale": [],
                    "failed": ["projection_worker_unavailable"],
                    "retryable": False,
                }
            return {"status": "skipped", "processed": [], "stale": [], "failed": []}
        result = self.projection_worker.process_commit_group(
            group_id,
            transaction_ids=transaction_ids,
        )
        return {
            "status": "completed" if not result["failed"] else "failed",
            "transaction_ids": list(transaction_ids),
            **result,
        }

    def _commit_consumer_operations(
        self,
        archive: SessionArchive,
        group_id: str,
        consumer: str,
        operations: list[ContextOperation],
    ) -> dict[str, Any]:
        self._stabilize_consumer_operations(archive, group_id, consumer, operations)
        return self._commit_or_describe(archive.user_id, operations)

    def _stabilize_consumer_operations(
        self,
        archive: SessionArchive,
        group_id: str,
        consumer: str,
        operations: list[ContextOperation],
    ) -> None:
        for index, operation in enumerate(operations):
            operation.operation_id = f"op_{stable_hash([group_id, consumer, index, operation.action.value, operation.target_uri], length=32)}"
            operation.created_at = archive.created_at
            operation.payload["commit_group_id"] = group_id
            operation.payload["commit_consumer"] = consumer

    def _enqueue_memory_proposal(self, archive: SessionArchive, exc: MemoryExtractionBackendError) -> None:
        self.queue_store.enqueue(
            QueueJob(
                job_id=f"memory_proposal_{archive.task_id}",
                queue_name="memory_proposal",
                action="retry_memory_proposal",
                target_uri=archive.archive_uri,
                payload={
                    "task_id": archive.task_id,
                    "tenant_id": self._tenant_id(archive),
                    "archive_digest": archive.archive_digest,
                    "manifest_digest": archive.manifest_digest,
                    "error_type": exc.error_type,
                },
            )
        )

    def _write_incomplete_outputs(
        self,
        archive: SessionArchive,
        group: CommitGroupStatus,
        abstract: str,
        overview: str,
        *,
        memory_diff: dict[str, Any],
    ) -> SessionCommitResult:
        pending = {"status": "pending", "operations": [], "operation_count": 0}
        memory_diff = {
            "archive_committed": True,
            "canonical_active_operation_count": 0,
            "pending_count": 0,
            "pending_persisted": False,
            **memory_diff,
        }
        self.archive_store.write_async_outputs(
            archive.archive_uri,
            abstract=abstract,
            overview=overview,
            memory_diff={"task_id": archive.task_id, "commit_group_id": group.group_id, **memory_diff},
            behavior_diff={"task_id": archive.task_id, **pending},
            action_policy_diff={"task_id": archive.task_id, **pending},
            context_diff={"task_id": archive.task_id, **pending},
            tenant_id=group.tenant_id,
            commit_group_status=group.to_dict(),
            complete=False,
        )
        return self._result(archive, group)

    def _result(self, archive: SessionArchive, group: CommitGroupStatus) -> SessionCommitResult:
        failed_consumers = [
            name for name, item in group.consumers.items() if item.status in {"failed", "dead_letter", "quarantine"}
        ]
        memory_result = dict(group.canonical_result or {})
        pending_count = int(memory_result.get("pending_count", 0) or 0)
        active_count = int(memory_result.get("canonical_active_operation_count", 0) or 0)
        pending_persisted = bool(memory_result.get("pending_persisted", False))
        if group.complete and pending_count:
            status = "done_with_pending"
        elif group.complete:
            status = "done"
        elif group.canonical_status == "completed" and failed_consumers:
            status = "derived_failed"
        elif group.canonical_status == "completed":
            status = "canonical_committed"
        else:
            status = "canonical_pending"
        terminal_failure = group.canonical_status in {"dead_letter", "quarantine"} or any(
            item.status in {"dead_letter", "quarantine"} for item in group.consumers.values()
        )
        retryable_failure = (group.canonical_status == "failed" and group.canonical_retryable) or any(
            item.status == "failed" and item.retryable for item in group.consumers.values()
        )
        return SessionCommitResult(
            task_id=archive.task_id,
            archive_uri=archive.archive_uri,
            status=status,
            done=group.complete,
            state=(
                SessionCommitState.DEAD_LETTER
                if terminal_failure
                else SessionCommitState.FAILED_RETRYABLE
                if retryable_failure
                else SessionCommitState.COMMITTED
                if group.canonical_status == "completed"
                else SessionCommitState.PROCESSING
            ),
            commit_group_id=group.group_id,
            canonical_committed=group.canonical_status == "completed",
            commit_group_status=group.to_dict(),
            archive_committed=True,
            canonical_active_operation_count=active_count,
            pending_count=pending_count,
            pending_persisted=pending_persisted,
        )

    def _tenant_id(self, archive: SessionArchive) -> str:
        metadata = dict(archive.metadata or {})
        scope = dict(metadata.get("scope", {}) or {})
        bound_tenant = str(self.archive_store.tenant_id)
        claimed_tenants = tuple(
            str(value) for value in (metadata.get("tenant_id"), scope.get("tenant_id")) if value not in (None, "")
        )
        if any(claimed != bound_tenant for claimed in claimed_tenants):
            raise PermissionError("session archive tenant does not match the bound archive store")
        return bound_tenant

    def _bind_archive_tenant(self, archive: SessionArchive) -> str:
        """Validate the trust boundary before I/O and materialize its tenant."""

        tenant_id = self._tenant_id(archive)
        metadata = dict(archive.metadata or {})
        metadata["tenant_id"] = tenant_id
        if "scope" in metadata:
            scope = dict(metadata.get("scope", {}) or {})
            scope["tenant_id"] = tenant_id
            metadata["scope"] = scope
        archive.metadata = metadata
        return tenant_id

    def _max_revision_from_diff(self, diff: dict[str, Any]) -> int | None:
        revisions = []
        for operation in diff.get("operations", []) or []:
            payload = operation.get("payload", {}).get("context_object") if isinstance(operation, dict) else None
            if isinstance(payload, dict):
                value = dict(payload.get("metadata", {}) or {}).get("revision")
                if value is not None:
                    revisions.append(int(value))
        return max(revisions) if revisions else None

    def _is_coding_agent(self, archive: SessionArchive) -> bool:
        connect = dict(archive.metadata.get("connect", {}) or {})
        return connect.get("connect_type") == "agent" and connect.get("run_mode") == "context_reduction"

    def _commit_or_describe(self, user_id: str, operations: list[ContextOperation]) -> dict:
        if self.committer is not None and operations:
            diff = self.committer.commit(user_id, operations)
            return self._diff_payload(diff, status="committed")
        if self.committer is not None:
            return {"status": "committed", "operations": [], "operation_count": 0}
        return {
            "status": "planned",
            "operations": [operation.to_dict() for operation in operations],
            "operation_count": len(operations),
        }

    def _diff_payload(self, diff: ContextDiff, status: str) -> dict:
        payload = diff.to_dict()
        payload["status"] = status
        payload["operation_count"] = len(diff.operations)
        return payload

    def _merge_diff_payloads(self, *payloads: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {"status": "committed"}
        for key in ("operations", "pending_operations", "rejected_operations"):
            by_id: dict[str, dict[str, Any]] = {}
            for payload in payloads:
                for operation in payload.get(key, []) or []:
                    if not isinstance(operation, dict):
                        continue
                    operation_id = str(operation.get("operation_id") or "")
                    if operation_id:
                        by_id.setdefault(operation_id, operation)
            merged[key] = list(by_id.values())
        merged["operation_count"] = len(merged["operations"])
        return merged

    def _backfill_canonical_effects(self, group_id: str, user_id: str) -> CommitGroupStatus:
        if self.committer is None:
            status = self.commit_group_store.load(group_id)
            if status is None:
                raise KeyError(f"unknown commit group: {group_id}")
            return status
        committed_effects = getattr(self.committer, "committed_memory_effect_diffs", None)
        if not callable(committed_effects):
            status = self.commit_group_store.load(group_id)
            if status is None:
                raise KeyError(f"unknown commit group: {group_id}")
            return status
        load_effects = cast(Callable[[str, str], list[ContextDiff]], committed_effects)
        for diff in load_effects(user_id, group_id):
            self.commit_group_store.record_canonical_effect(group_id, diff.to_dict())
        status = self.commit_group_store.load(group_id)
        if status is None:
            raise KeyError(f"unknown commit group: {group_id}")
        return status

    def _recover_pending_memory_effects(self, user_id: str, group_id: str) -> None:
        if self.committer is None:
            return
        recover_canonical = getattr(self.committer, "recover_pending_canonical", None)
        if callable(recover_canonical):
            recover_canonical(
                user_id,
                commit_group_id=group_id or None,
            )
        if not group_id:
            return
        recover_regular = getattr(self.committer, "recover_pending_regular_memory", None)
        if callable(recover_regular):
            recover_regular(
                user_id,
                commit_group_id=group_id,
            )

    def _merge_canonical_effects(
        self,
        group_id: str,
        user_id: str,
        current: dict[str, Any],
    ) -> dict[str, Any]:
        if self.committer is None:
            return current
        status = self.commit_group_store.load(group_id)
        if status is None:
            raise KeyError(f"unknown commit group: {group_id}")
        diffs = [self._context_diff_from_payload(payload) for payload in status.canonical_effects.values()]
        if any(current.get(key) for key in ("operations", "pending_operations", "rejected_operations")):
            diffs.append(self._context_diff_from_payload(current))
        if not diffs:
            return current
        combine = getattr(self.committer, "combine_committed_diffs", None)
        if callable(combine):
            combine_diffs = cast(Callable[[str, list[ContextDiff]], ContextDiff], combine)
            combined = combine_diffs(user_id, diffs)
            return self._diff_payload(combined, status="committed")
        return self._merge_diff_payloads(*(self._diff_payload(diff, "committed") for diff in diffs))

    def _context_diff_from_payload(self, payload: dict[str, Any]) -> ContextDiff:
        return ContextDiff(
            user_id=str(payload["user_id"]),
            operations=[ContextOperation.from_dict(item) for item in payload.get("operations", []) or []],
            pending_operations=[
                ContextOperation.from_dict(item) for item in payload.get("pending_operations", []) or []
            ],
            rejected_operations=[
                ContextOperation.from_dict(item) for item in payload.get("rejected_operations", []) or []
            ],
            diff_id=str(payload.get("diff_id") or ""),
            created_at=str(payload.get("created_at") or ""),
            schema_version=str(payload.get("schema_version") or "context_diff_v1"),
        )

    def _partial_memory_result(self, group: CommitGroupStatus) -> dict[str, Any]:
        if group.canonical_result:
            return {
                **group.canonical_result,
                "status": "partial_failed",
                "error": group.canonical_last_error,
            }
        if group.canonical_effects:
            merged = self._merge_diff_payloads(*group.canonical_effects.values())
            merged.update(
                {
                    "user_id": group.user_id,
                    "status": "partial_failed",
                    "error": group.canonical_last_error,
                }
            )
            return self._memory_commit_metadata(merged, [])
        return {
            "status": "failed",
            "error": group.canonical_last_error,
            "operation_count": 0,
            "operations": [],
        }

    def _memory_commit_metadata(
        self,
        payload: dict[str, Any],
        planned_operations: list[ContextOperation],
    ) -> dict[str, Any]:
        planned_pending = {
            str(operation.target_uri or operation.payload.get("pending_proposal_id") or operation.operation_id)
            for operation in planned_operations
            if operation.payload.get("canonical_pending_proposal") is True
        }
        committed_pending = {
            str(operation.get("target_uri") or operation.get("payload", {}).get("pending_proposal_id") or "")
            for operation in payload.get("operations", []) or []
            if isinstance(operation, dict) and operation.get("payload", {}).get("canonical_pending_proposal") is True
        }
        if self.committer is not None and planned_pending - committed_pending:
            raise RuntimeError("canonical pending proposals were not durably committed")
        outstanding_pending = self._outstanding_pending_uris(
            planned_operations,
            committed_pending,
        )
        active_operations = [
            operation
            for operation in payload.get("operations", []) or []
            if isinstance(operation, dict)
            and operation.get("payload", {}).get("canonical_memory") is True
            and dict(dict(operation.get("payload", {}).get("context_object", {}) or {}).get("metadata", {}) or {}).get(
                "canonical_kind"
            )
            == "claim"
            and dict(dict(operation.get("payload", {}).get("context_object", {}) or {}).get("metadata", {}) or {}).get(
                "state"
            )
            == "ACTIVE"
        ]
        return {
            **payload,
            "archive_committed": True,
            "canonical_active_operation_count": len(active_operations),
            "pending_count": len(outstanding_pending),
            "pending_persisted": bool(outstanding_pending) and self.committer is not None,
        }

    def _outstanding_pending_uris(
        self,
        planned_operations: list[ContextOperation],
        committed_pending: set[str],
    ) -> set[str]:
        outstanding_states = {
            LifecycleState.PENDING.value,
            LifecycleState.RETRYABLE.value,
            LifecycleState.CONFIRMED.value,
        }
        if self.committer is not None:
            outstanding: set[str] = set()
            for uri in committed_pending:
                try:
                    obj = read_committed_pending(
                        self.committer.source_store,
                        uri,
                        getattr(self.committer, "relation_store", None),
                    ).object
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as exc:
                    raise RuntimeError("committed pending proposal has no valid lifecycle proof") from exc
                metadata = dict(obj.metadata or {})
                if (
                    metadata.get("canonical_kind") == "pending_proposal"
                    and obj.lifecycle_state.value in outstanding_states
                ):
                    outstanding.add(uri)
            return outstanding

        planned_states: dict[str, str] = {}
        for operation in planned_operations:
            if operation.payload.get("canonical_pending_proposal") is not True:
                continue
            uri = str(operation.target_uri or operation.payload.get("pending_proposal_id") or operation.operation_id)
            object_payload = operation.payload.get("context_object")
            object_state = str(object_payload.get("lifecycle_state") or "") if isinstance(object_payload, dict) else ""
            planned_states[uri] = str(operation.payload.get("pending_lifecycle_state") or object_state)
        return {uri for uri, state in planned_states.items() if state in outstanding_states}
