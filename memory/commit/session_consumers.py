"""Session 提交的 Memory 与普通派生消费者执行。"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable, Sequence
from typing import Any

from infrastructure.context.layers.generator import l0_abstract, l1_overview
from infrastructure.store.memory.control_store import (
    document_intent_id,
)
from infrastructure.store.session.commit_group import (
    CONSUMERS,
    CommitGroupStatus,
)
from memory.commit.document_commit import (
    DocumentCommitConflict,
    DocumentCommitResult,
)
from memory.commit.model.session import (
    SessionCommitResult,
    SessionCommitState,
)
from memory.commit.session_commit_types import (
    ConsumerLeaseBusy,
    ConsumerTerminalError,
)
from memory.commit.session_recovery import _SessionCommitRecovery
from memory.core.model import DocumentEditPlan, MemoryEditProposal
from memory.ports.document_store import DocumentConflictError
from pre.session import SessionArchive
from transaction.model.context_diff import ContextDiff
from transaction.model.context_operation import ContextOperation


class _SessionCommitConsumers(_SessionCommitRecovery):
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
        """恢复已经提交 Source、但尚未记录提交组副作用时崩溃的任务。"""

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
