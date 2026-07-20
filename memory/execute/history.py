"""记忆文档历史查询和指定版本恢复操作。"""

from __future__ import annotations

import hashlib

from foundation.identity import LocalUserContext
from foundation.integrity import canonical_json
from infrastructure.store.memory.control_store import document_intent_id
from infrastructure.store.memory.revision_store import DocumentRevisionRecord
from memory.core.model import AbsentPath, DocumentEditKind, DocumentEditPlan, PresentPath, UnsafePath
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.execute.base import MemoryCommandBase, _assert_restore_expected
from memory.execute.contracts import DocumentEditResult, MemoryHistoryResult, MemoryRevisionInfo
from memory.ports.document_store import DocumentConflictError, DocumentNotFoundError


class HistoryOperation(MemoryCommandBase):
    """读取受保护的版本记录，并将恢复表达为一次新的 CAS 修改。"""

    def list_memory_history(
        self,
        document_uri: str,
        *,
        caller: LocalUserContext,
    ) -> MemoryHistoryResult:
        self._require_ready()
        owner, document_id = self._bind_document_uri(document_uri, caller)
        self.erase_store.assert_mutation_allowed(caller.tenant_id, owner, document_id)
        records = self.revision_store.list_revisions(caller.tenant_id, owner, document_id)
        if not records:
            raise DocumentNotFoundError("memory document has no retained history")
        latest = records[-1]
        return MemoryHistoryResult(
            document_uri=document_uri,
            document_id=document_id,
            document_kind=MemoryDocumentPathPolicy.kind_for(latest.relative_path).value,
            relative_path=latest.relative_path,
            revisions=tuple(_revision_info(record) for record in records),
        )

    def restore_memory_revision(
        self,
        document_uri: str,
        revision: int,
        expected_digest: str,
        *,
        caller: LocalUserContext,
    ) -> DocumentEditResult:
        self._require_ready()
        owner, document_id = self._bind_document_uri(document_uri, caller)
        self.erase_store.assert_mutation_allowed(caller.tenant_id, owner, document_id)
        if revision <= 0:
            raise ValueError("revision must be positive")
        control = self.control_store.load_control(caller.tenant_id, owner, document_id)
        if control is None:
            raise DocumentNotFoundError("memory document control state does not exist")
        state = self.document_store.read_state(caller.tenant_id, owner, control.relative_path)
        if isinstance(state, UnsafePath):
            raise DocumentConflictError("memory document path is unsafe")
        edit_kind = DocumentEditKind.CREATE if isinstance(state, AbsentPath) else DocumentEditKind.UPDATE
        evidence_digest = hashlib.sha256(
            canonical_json(["RESTORE", document_uri, revision, expected_digest]).encode()
        ).hexdigest()
        plan = DocumentEditPlan(
            idempotency_key="restore:"
            + hashlib.sha256(canonical_json([document_uri, revision, expected_digest]).encode()).hexdigest(),
            tenant_id=caller.tenant_id,
            owner_user_id=owner,
            edit_kind=edit_kind,
            expected_state=state,
            evidence_digest=evidence_digest,
            edit_summary=f"restore retained document revision {revision}",
            document_id=document_id,
            relative_path=control.relative_path,
            expected_registration_document_id=document_id if isinstance(state, PresentPath) else "",
        )
        evidence_reference = f"restore-revision:{revision}:sha256:{evidence_digest}"
        idempotency_digest = hashlib.sha256(plan.idempotency_key.encode()).hexdigest()
        intent_id = document_intent_id(
            plan.tenant_id,
            plan.owner_user_id,
            plan.document_id,
            idempotency_digest,
        )
        existing = self.control_store.load_intent(plan.tenant_id, plan.owner_user_id, intent_id)
        if existing is not None:
            if (
                existing.evidence_digest != plan.evidence_digest
                or existing.actor_binding != self._actor_binding(caller)
                or existing.evidence_reference != evidence_reference
            ):
                raise DocumentConflictError("restore replay is detached from its durable intent")
            result = self.committer.recover_intent(plan.tenant_id, plan.owner_user_id, intent_id)
        else:
            _assert_restore_expected(state, expected_digest)
            result = self.committer.restore_revision(
                plan,
                revision=revision,
                actor_binding=self._actor_binding(caller),
                evidence_reference=evidence_reference,
            )
        return DocumentEditResult(**self._result_fields(plan, result))


def _revision_info(record: DocumentRevisionRecord) -> MemoryRevisionInfo:
    return MemoryRevisionInfo(
        document_revision=record.logical_revision,
        projection_generation=record.projection_generation,
        edit_kind=record.edit_kind.value,
        relative_path=record.relative_path,
        source_digest=record.raw_sha256,
        state=record.state,
        created_at=record.created_at,
        restorable=bool(record.content_blob_digest),
    )


__all__ = ["HistoryOperation"]
