"""多文档合并事务的只向前协调器。"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import replace

from foundation.clock import utc_now
from infrastructure.store.memory.control_store import document_intent_id
from memory.commit.document_commit import DocumentCommitResult, MemoryDocumentCommitter
from memory.core.model import DocumentEditKind, DocumentEditPlan, PresentPath
from memory.core.structure.frontmatter import validate_document_id
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.ports.consolidation import (
    _MAX_SOURCES,
    ConsolidationFaultHook,
    ConsolidationInputRequired,
    ConsolidationIntegrityError,
    ConsolidationProjectionReader,
    ConsolidationRecoveryReport,
    ConsolidationResult,
    ConsolidationSagaRecord,
    ConsolidationSagaStore,
    ConsolidationSource,
    ConsolidationStatus,
    _coerce_int,
    _target_plan_digest,
    consolidation_saga_id,
)
from memory.ports.document_store import DocumentConflictError


class MemoryDocumentConsolidator:
    """只向前推进数量受限的多文档合并事务。"""

    def __init__(
        self,
        committer: MemoryDocumentCommitter,
        projection_store: ConsolidationProjectionReader,
        *,
        saga_store: ConsolidationSagaStore,
        clock: Callable[[], str] = utc_now,
        test_hook: ConsolidationFaultHook | None = None,
    ) -> None:
        self.committer = committer
        self.projection_store = projection_store
        self.saga_store = saga_store
        self.clock = clock
        self.test_hook = test_hook

    def consolidate(
        self,
        target_plan: DocumentEditPlan,
        sources: Sequence[ConsolidationSource],
        *,
        idempotency_key: str,
        actor_binding: str,
    ) -> ConsolidationResult:
        """提交并确认目标投影；来源 Markdown 始终保留。"""

        prepared = self._prepare_record(
            target_plan,
            sources,
            idempotency_key=idempotency_key,
            actor_binding=actor_binding,
        )
        with self.saga_store.lock(prepared.tenant_id, prepared.owner_user_id, prepared.saga_id):
            record = self.saga_store.load(prepared.tenant_id, prepared.owner_user_id, prepared.saga_id)
            if record is None:
                record = self.saga_store.create(prepared)
                self._notify("after_saga_checkpoint", record)
            elif record.identity_digest != prepared.identity_digest:
                raise ConsolidationIntegrityError("consolidation retry changed its immutable inputs")
            return self._advance(record, target_plan=target_plan)

    def resume(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        saga_id: str,
        actor_binding: str,
    ) -> ConsolidationResult:
        """从已经耐久准备目标 Intent 的阶段继续 Saga。"""

        with self.saga_store.lock(tenant_id, owner_user_id, saga_id):
            record = self.saga_store.load(tenant_id, owner_user_id, saga_id)
            if record is None:
                raise ConsolidationInputRequired("consolidation journal does not exist")
            if record.actor_binding != actor_binding:
                raise PermissionError("consolidation recovery actor binding differs from its journal")
            return self._advance(record, target_plan=None)

    def resume_all(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        limit: int = 1_000,
    ) -> ConsolidationRecoveryReport:
        """使用已封存的操作者绑定，在上限内恢复所有待处理 Saga。

        如果 PREPARED Saga 没有耐久目标 Intent，仅凭无正文日志无法重建；
        此时将其报告为等待输入并保持不动。
        """

        pending = self.saga_store.list_pending(tenant_id, owner_user_id, limit=limit)
        completed: list[str] = []
        awaiting_projection: list[str] = []
        awaiting_input: list[str] = []
        for snapshot in pending:
            with self.saga_store.lock(tenant_id, owner_user_id, snapshot.saga_id):
                record = self.saga_store.load(tenant_id, owner_user_id, snapshot.saga_id)
                if record is None or record.status == ConsolidationStatus.COMPLETED:
                    continue
                try:
                    result = self._advance(record, target_plan=None)
                except ConsolidationInputRequired:
                    awaiting_input.append(record.saga_id)
                    continue
                if result.status == ConsolidationStatus.COMPLETED:
                    completed.append(record.saga_id)
                elif not result.target_projection_confirmed:
                    awaiting_projection.append(record.saga_id)
        return ConsolidationRecoveryReport(
            examined=len(pending),
            completed_saga_ids=tuple(completed),
            awaiting_projection_saga_ids=tuple(awaiting_projection),
            awaiting_input_saga_ids=tuple(awaiting_input),
        )

    def _advance(
        self,
        record: ConsolidationSagaRecord,
        *,
        target_plan: DocumentEditPlan | None,
    ) -> ConsolidationResult:
        if record.status == ConsolidationStatus.COMPLETED:
            return self._result(record)

        if record.target_projection_generation == 0:
            target_result = self._commit_target(record, target_plan)
            self._notify("after_target_commit", record)
            control = target_result.control or self.committer.control_store.load_control(
                record.tenant_id,
                record.owner_user_id,
                record.target_document_id,
            )
            if control is None or control.status != "present" or control.raw_sha256 != record.target_source_digest:
                raise DocumentConflictError("consolidation target commit did not install its exact Markdown")
            record = self.saga_store.save(
                replace(
                    record,
                    status=ConsolidationStatus.AWAITING_TARGET_PROJECTION,
                    target_projection_generation=control.projection_generation,
                    updated_at=self.clock(),
                )
            )
            self._notify("after_target_checkpoint", record)

        self._require_target_live(record)
        if not self._target_projection_matches(record):
            return self._result(record)

        if not record.target_projection_confirmed_at:
            record = self.saga_store.save(
                replace(
                    record,
                    status=ConsolidationStatus.COMPLETED,
                    target_projection_confirmed_at=self.clock(),
                    updated_at=self.clock(),
                )
            )
            self._notify("after_projection_checkpoint", record)
        return self._result(record)

    def _commit_target(
        self,
        record: ConsolidationSagaRecord,
        target_plan: DocumentEditPlan | None,
    ) -> DocumentCommitResult:
        existing = self.committer.control_store.load_intent(
            record.tenant_id,
            record.owner_user_id,
            record.target_intent_id,
        )
        if existing is not None:
            return self.committer.recover_intent(
                record.tenant_id,
                record.owner_user_id,
                record.target_intent_id,
            )
        if target_plan is None:
            raise ConsolidationInputRequired(
                "target plan bytes were never durably prepared; resubmit the exact consolidation request"
            )
        if _target_plan_digest(target_plan) != record.target_plan_digest:
            raise ConsolidationIntegrityError("resubmitted target plan differs from its saga journal")
        return self.committer.commit(
            target_plan,
            actor_binding=record.actor_binding,
            evidence_reference=f"consolidation:{record.saga_id}:target",
        )

    def _require_target_live(self, record: ConsolidationSagaRecord) -> None:
        control = self.committer.control_store.load_control(
            record.tenant_id,
            record.owner_user_id,
            record.target_document_id,
        )
        live = self.committer.document_store.read_state(
            record.tenant_id,
            record.owner_user_id,
            record.target_relative_path,
        )
        if (
            control is None
            or control.status != "present"
            or control.raw_sha256 != record.target_source_digest
            or control.projection_generation != record.target_projection_generation
            or not isinstance(live, PresentPath)
            or live.raw_sha256 != record.target_source_digest
        ):
            raise DocumentConflictError("consolidation target changed after commit; redundant sources were preserved")

    def _target_projection_matches(self, record: ConsolidationSagaRecord) -> bool:
        state = self.projection_store.get_memory_document_projection_state(
            tenant_id=record.tenant_id,
            owner_user_id=record.owner_user_id,
            document_id=record.target_document_id,
        )
        if state is None:
            return False
        return (
            str(state.get("tenant_id") or "") == record.tenant_id
            and str(state.get("owner_user_id") or "") == record.owner_user_id
            and str(state.get("document_id") or "") == record.target_document_id
            and str(state.get("source_digest") or "") == record.target_source_digest
            and _coerce_int(
                state.get("projection_generation") or 0,
                "serving projection generation",
            )
            == record.target_projection_generation
            and str(state.get("projection_status") or "") == "PROJECTED"
            and not str(state.get("deletion_status") or "")
        )

    def _prepare_record(
        self,
        target_plan: DocumentEditPlan,
        sources: Sequence[ConsolidationSource],
        *,
        idempotency_key: str,
        actor_binding: str,
    ) -> ConsolidationSagaRecord:
        if target_plan.edit_kind not in {DocumentEditKind.CREATE, DocumentEditKind.UPDATE}:
            raise ValueError("consolidation target must be a CREATE or UPDATE plan")
        if target_plan.after_bytes is None:
            raise ValueError("consolidation target plan requires exact after bytes")
        tenant = MemoryDocumentPathPolicy.trusted_segment(target_plan.tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(target_plan.owner_user_id, "owner_user_id")
        target = validate_document_id(target_plan.document_id)
        path = MemoryDocumentPathPolicy.normalize_relative_path(target_plan.relative_path)
        key = str(idempotency_key or "").strip()
        if not key or len(key) > 512:
            raise ValueError("consolidation idempotency key must be non-empty and bounded")
        actor = str(actor_binding or "").strip()
        if not actor or len(actor) > 512:
            raise ValueError("consolidation actor binding must be non-empty and bounded")
        bounded_sources = tuple(sources)
        if len(bounded_sources) > _MAX_SOURCES:
            raise ValueError("consolidation source count exceeds its bound")
        idempotency_digest = hashlib.sha256(key.encode()).hexdigest()
        saga_id = consolidation_saga_id(tenant, owner, idempotency_digest)
        target_key_digest = hashlib.sha256(target_plan.idempotency_key.encode()).hexdigest()
        now = self.clock()
        return ConsolidationSagaRecord(
            saga_id=saga_id,
            identity_digest=None,
            idempotency_digest=idempotency_digest,
            tenant_id=tenant,
            owner_user_id=owner,
            actor_binding=actor,
            target_document_id=target,
            target_relative_path=path,
            target_source_digest=hashlib.sha256(target_plan.after_bytes).hexdigest(),
            target_plan_digest=_target_plan_digest(target_plan),
            target_intent_id=document_intent_id(tenant, owner, target, target_key_digest),
            sources=bounded_sources,
            status=ConsolidationStatus.PREPARED,
            target_projection_generation=0,
            target_projection_confirmed_at="",
            created_at=now,
            updated_at=now,
        )

    @staticmethod
    def _result(record: ConsolidationSagaRecord) -> ConsolidationResult:
        return ConsolidationResult(
            saga_id=record.saga_id,
            status=record.status,
            target_document_id=record.target_document_id,
            target_projection_generation=record.target_projection_generation,
            target_projection_confirmed=bool(record.target_projection_confirmed_at),
            preserved_source_document_ids=tuple(source.document_id for source in record.sources),
        )

    def _notify(self, stage: str, record: ConsolidationSagaRecord) -> None:
        if self.test_hook is not None:
            self.test_hook(stage, record)


__all__ = ["MemoryDocumentConsolidator"]
