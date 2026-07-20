"""Markdown 记忆文档的耐久 CAS 提交入口。"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from foundation.clock import utc_now
from infrastructure.store.contracts.path_lock import PathLock
from infrastructure.store.contracts.queue import QueueStore
from infrastructure.store.memory.control_store import (
    MemoryDocumentControlStore,
    document_intent_id,
)
from infrastructure.store.memory.revision_store import (
    MemoryDocumentRevisionStore,
)
from memory.core.model import (
    DocumentEditKind,
    DocumentEditPlan,
)
from memory.core.structure.frontmatter import validate_document_id
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.ports.document_store import (
    MemoryDocumentStore,
)
from memory.ports.erase import DocumentEraseStore

if TYPE_CHECKING:
    pass
from memory.commit.document_commit_types import (
    DocumentCommitConflict,
    DocumentCommitResult,
    DocumentRecoveryReport,
    FaultHook,
    _bounded_text,
    _sha256_digest,
)
from memory.commit.document_publication import _DocumentCommitPublication


class MemoryDocumentCommitter(_DocumentCommitPublication):
    """协调单文档准备、安装、发布和只向前恢复。"""

    def __init__(
        self,
        document_store: MemoryDocumentStore,
        control_store: MemoryDocumentControlStore,
        revision_store: MemoryDocumentRevisionStore,
        projection_queue: QueueStore,
        *,
        erasure_store: DocumentEraseStore,
        path_lock: PathLock | None = None,
        test_hook: FaultHook | None = None,
        clock: Callable[[], str] = utc_now,
    ) -> None:
        self.document_store = document_store
        self.control_store = control_store
        self.revision_store = revision_store
        self.projection_queue = projection_queue
        self.path_lock = path_lock
        self.test_hook = test_hook
        self.clock = clock
        self.erasure_store = erasure_store

    def commit(
        self,
        plan: DocumentEditPlan,
        *,
        actor_binding: str,
        evidence_reference: str,
    ) -> DocumentCommitResult:
        """准备并只向前推进一次精确文档 CAS。"""

        return self._commit(
            plan,
            actor_binding=actor_binding,
            evidence_reference=evidence_reference,
            restore_deletion_generation=0,
        )

    def preflight_adoption_create(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> None:
        """在收养未受管正文之前绑定可信根身份。"""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        identifier = validate_document_id(document_id)
        with (
            self._document_identity_lock(
                tenant,
                owner,
                identifier,
            ),
            self.control_store.root_identity_lock(tenant, owner) as root_guard,
        ):
            self._preflight_new_intent_root(
                tenant,
                owner,
                edit_kind=DocumentEditKind.CREATE,
                allow_initial_publication=True,
                root_guard=root_guard,
            )

    def verify_adoption_root(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> None:
        """校验已经预检的收养重试，不补写任何授权状态。"""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        identifier = validate_document_id(document_id)
        with self._document_identity_lock(tenant, owner, identifier):
            self._verify_existing_root_identity(tenant, owner)

    def _commit(
        self,
        plan: DocumentEditPlan,
        *,
        actor_binding: str,
        evidence_reference: str,
        restore_deletion_generation: int,
    ) -> DocumentCommitResult:
        """携带不可伪造显式恢复标记的内部提交入口。"""

        actor = _bounded_text(actor_binding, "actor_binding", 512)
        evidence = _bounded_text(evidence_reference, "evidence_reference", 2048)
        summary = _bounded_text(plan.edit_summary, "edit_summary", 500)
        idempotency_key = _bounded_text(plan.idempotency_key, "idempotency_key", 512)
        evidence_digest = _sha256_digest(plan.evidence_digest, "evidence_digest")
        tenant = MemoryDocumentPathPolicy.trusted_segment(plan.tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(plan.owner_user_id, "owner_user_id")
        document_id = validate_document_id(plan.document_id)
        if plan.expected_registration_document_id and plan.expected_registration_document_id != document_id:
            raise ValueError("expected registration is detached from the plan document identity")

        idempotency_digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
        intent_id = document_intent_id(tenant, owner, document_id, idempotency_digest)
        prepared = self._prepare_plan(plan, tenant=tenant, owner=owner, document_id=document_id)
        try:
            with self._document_identity_lock(tenant, owner, document_id):
                outcome, recovered = self._prepare_locked(
                    prepared,
                    tenant=tenant,
                    owner=owner,
                    document_id=document_id,
                    intent_id=intent_id,
                    idempotency_digest=idempotency_digest,
                    actor=actor,
                    evidence=evidence,
                    evidence_digest=evidence_digest,
                    summary=summary,
                    restore_deletion_generation=restore_deletion_generation,
                )
            if isinstance(outcome, DocumentCommitResult):
                return outcome
            return self._resume_existing(outcome, recovered=recovered)
        except BaseException as exc:
            self._enqueue_retryable_recovery(
                tenant,
                owner,
                document_id,
                intent_id,
                exc,
            )
            raise


__all__ = [
    "DocumentCommitConflict",
    "DocumentCommitResult",
    "DocumentRecoveryReport",
    "MemoryDocumentCommitter",
]
