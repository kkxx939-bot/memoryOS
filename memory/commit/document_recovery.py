"""单文档提交、外部变更和历史版本的恢复入口。"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import TYPE_CHECKING

from infrastructure.store.memory.control_store import (
    DocumentCommitIntent,
    DocumentControlIntegrityError,
    DocumentControlRecord,
    DocumentDeletionStatus,
    DocumentIntentStatus,
    DocumentPathEffect,
    document_intent_id,
)
from memory.core.model import (
    ABSENT,
    DocumentEditKind,
    DocumentEditPlan,
    PresentPath,
)
from memory.core.structure.frontmatter import matches_adopted_source, validate_document_id
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.ports.document_store import (
    DocumentNotFoundError,
)

if TYPE_CHECKING:
    from infrastructure.store.memory.scanner import ExternalDocumentChange
from memory.commit.document_commit_types import (
    DocumentCommitConflict,
    DocumentCommitResult,
    DocumentRecoveryReport,
    _bounded_text,
    _sha256_digest,
)
from memory.commit.document_prepare import _DocumentCommitPreparation


class _DocumentCommitRecovery(_DocumentCommitPreparation):
    def recover_intent(
        self,
        tenant_id: str,
        owner_user_id: str,
        intent_id: str,
    ) -> DocumentCommitResult:
        intent = self.control_store.load_intent(tenant_id, owner_user_id, intent_id)
        if intent is None:
            raise DocumentNotFoundError("document commit intent does not exist")
        self.erasure_store.assert_mutation_allowed(tenant_id, owner_user_id, intent.document_id)
        return self._resume_existing(intent, recovered=True)

    def recover_all(self, tenant_id: str, owner_user_id: str) -> DocumentRecoveryReport:
        # 新 Blob 发布和 PREPARED Intent 创建持有同一所有者锁，
        # 因此 GC 不会与 Blob 先于 Intent 出现的短暂窗口竞争。
        with self.control_store.root_identity_lock(tenant_id, owner_user_id):
            self.revision_store.prune_unreferenced_blobs(
                tenant_id,
                owner_user_id,
                self.control_store.intents(tenant_id, owner_user_id),
            )
        completed: list[DocumentCommitResult] = []
        conflicted: list[str] = []
        for intent in self.control_store.incomplete_intents(tenant_id, owner_user_id):
            try:
                completed.append(self._resume_existing(intent, recovered=True))
            except DocumentCommitConflict:
                conflicted.append(intent.intent_id)
        return DocumentRecoveryReport(tuple(completed), tuple(conflicted))

    def record_external_change(
        self,
        change: ExternalDocumentChange,
        *,
        actor_binding: str | None = None,
        evidence_reference: str | None = None,
        evidence_digest: str | None = None,
        idempotency_key: str | None = None,
        edit_summary: str | None = None,
    ) -> DocumentCommitResult | None:
        """记录扫描器确认的外部编辑，但不重写其正文。

        已准备 Intent 的精确 after 状态已经存在于当前文件系统中。
        正常恢复会观察到 ``live == after``，只推进不含正文的事件、
        修订、控制记录和队列尾部；后续第三状态仍由常规恢复保护保留。
        """

        tenant = MemoryDocumentPathPolicy.trusted_segment(change.tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(change.owner_user_id, "owner_user_id")
        document_id = validate_document_id(change.document_id)
        change_kind = change.change_kind.value
        old_path = str(change.old_relative_path or "")
        new_path = str(change.new_relative_path or "")
        scan_id = str(change.scan_generation_id or "")
        if not scan_id:
            raise ValueError("external document change requires a scan generation")
        if (
            actor_binding is None
            and evidence_reference is None
            and evidence_digest is None
            and idempotency_key is None
            and edit_summary is None
            and change_kind == DocumentEditKind.CREATE.value
            and self.control_store.load_control(tenant, owner, document_id) is None
        ):
            receipt = self.control_store.load_adoption_receipt_for_document(
                tenant,
                owner,
                document_id,
            )
            if receipt is not None:
                if old_path or new_path != receipt.relative_path:
                    raise DocumentCommitConflict("scanner CREATE is detached from its durable adoption receipt path")
                raw = self.document_store.read_raw(
                    tenant,
                    owner,
                    document_id=document_id,
                )
                if hashlib.sha256(raw).hexdigest() != str(change.after_raw_digest or "") or not matches_adopted_source(
                    raw,
                    receipt.document_id,
                    receipt.expected_raw_sha256,
                ):
                    raise DocumentCommitConflict("scanner CREATE does not match the exact durable adoption receipt")
                actor_binding = receipt.actor_binding
                evidence_reference = receipt.evidence_reference
                evidence_digest = receipt.evidence_digest
                idempotency_key = receipt.idempotency_key
                edit_summary = receipt.edit_summary
        default_identity = "\x1f".join(
            (
                "external-document-change-v1",
                tenant,
                owner,
                document_id,
                scan_id,
                change_kind,
                old_path,
                new_path,
                str(change.before_raw_digest or ""),
                str(change.after_raw_digest or ""),
            )
        )
        stable_key = (
            _bounded_text(idempotency_key, "idempotency_key", 512) if idempotency_key is not None else default_identity
        )
        idempotency_digest = hashlib.sha256(stable_key.encode()).hexdigest()
        intent_id = document_intent_id(tenant, owner, document_id, idempotency_digest)
        actor = _bounded_text(
            actor_binding if actor_binding is not None else "external-editor:stable-full-scan",
            "actor_binding",
            512,
        )
        evidence = _bounded_text(
            evidence_reference if evidence_reference is not None else f"scan-generation:{scan_id}",
            "evidence_reference",
            2048,
        )
        evidence_hash = _sha256_digest(
            evidence_digest if evidence_digest is not None else hashlib.sha256(default_identity.encode()).hexdigest(),
            "evidence_digest",
        )
        summary = _bounded_text(
            edit_summary if edit_summary is not None else f"external Markdown {change_kind}",
            "edit_summary",
            500,
        )
        existing = self.control_store.load_intent(tenant, owner, intent_id)
        if existing is not None:
            self._assert_external_request_matches(
                existing,
                change_kind=change_kind,
                old_path=old_path,
                new_path=new_path,
                actor_binding=actor,
                evidence_reference=evidence,
                evidence_digest=evidence_hash,
                edit_summary=summary,
                before_digest=str(change.before_raw_digest or ""),
                after_digest=str(change.after_raw_digest or ""),
                match_raw_digests=idempotency_key is None,
            )
            return self._resume_existing(existing, recovered=True)
        with (
            self._document_identity_lock(
                tenant,
                owner,
                document_id,
            ),
            self.control_store.root_identity_lock(tenant, owner) as root_guard,
        ):
            self.erasure_store.assert_mutation_allowed(tenant, owner, document_id)
            control = self.control_store.load_control(tenant, owner, document_id)
            barrier = self.control_store.load_publication_barrier(tenant, owner, document_id)
            if barrier is not None and barrier.status is DocumentDeletionStatus.HARD_ERASED:
                raise DocumentCommitConflict("hard-erased document identity cannot be adopted")
            if barrier is not None and (
                control is None
                or control.status != "present"
                or control.restored_from_deletion_generation != barrier.deletion_generation
                or control.projection_generation <= barrier.deletion_generation
            ):
                raise DocumentCommitConflict(
                    "scanner cannot adopt bytes blocked by a durable deletion publication barrier"
                )
            if control is not None and control.status == "present":
                if control.raw_sha256 == str(change.after_raw_digest or "") and control.relative_path == (
                    new_path or old_path
                ):
                    self._verify_existing_root_identity(tenant, owner)
                    return None
                if control.raw_sha256 != str(change.before_raw_digest or "") or control.relative_path != old_path:
                    raise DocumentCommitConflict(
                        "external scan is detached from durable document control; live Markdown was preserved"
                    )
            elif control is not None:
                raise DocumentCommitConflict("soft-forgotten document requires an explicit restore")

            effects, edit_kind, raw = self._external_effects(
                tenant,
                owner,
                document_id,
                change_kind=change_kind,
                old_path=old_path,
                new_path=new_path,
                before_digest=str(change.before_raw_digest or ""),
                control=control,
            )
            if effects is None:
                return None
            adoption_receipt = self.control_store.load_adoption_receipt_for_document(
                tenant,
                owner,
                document_id,
            )
            self._preflight_new_intent_root(
                tenant,
                owner,
                edit_kind=edit_kind,
                allow_initial_publication=(edit_kind is DocumentEditKind.CREATE and adoption_receipt is None),
                root_guard=root_guard,
            )
            latest_revision = self.revision_store.latest_revision(tenant, owner, document_id)
            latest_record = (
                self.revision_store.load_revision(tenant, owner, document_id, latest_revision)
                if latest_revision
                else None
            )
            logical_revision = max(latest_revision, control.logical_revision if control is not None else 0) + 1
            projection_generation = (
                max(
                    latest_record.projection_generation if latest_record is not None else 0,
                    control.projection_generation if control is not None else 0,
                )
                + 1
            )
            event_digest = hashlib.sha256(f"{intent_id}:event".encode()).hexdigest()
            job_digest = hashlib.sha256(f"{intent_id}:projection".encode()).hexdigest()
            now = self.clock()
            content_digest = hashlib.sha256(raw).hexdigest() if raw else ""
            intent = DocumentCommitIntent(
                intent_id=intent_id,
                idempotency_digest=idempotency_digest,
                identity_digest=None,
                tenant_id=tenant,
                owner_user_id=owner,
                document_id=document_id,
                edit_kind=edit_kind,
                effects=effects,
                after_blob_digest=content_digest if edit_kind is not DocumentEditKind.DELETE else "",
                revision_blob_digest=content_digest,
                revision_blob_role="before_delete" if edit_kind is DocumentEditKind.DELETE else "after",
                logical_revision=logical_revision,
                projection_generation=projection_generation,
                event_id=f"memchg_{event_digest}",
                projection_job_id=f"memory_projection_{job_digest}",
                old_relative_path=old_path if edit_kind is not DocumentEditKind.CREATE else "",
                new_relative_path=new_path or (old_path if edit_kind is DocumentEditKind.UPDATE else ""),
                actor_binding=actor,
                evidence_reference=evidence,
                evidence_digest=evidence_hash,
                edit_summary=summary,
                status=DocumentIntentStatus.PREPARED,
                created_at=now,
                updated_at=now,
                restored_from_deletion_generation=(
                    control.restored_from_deletion_generation
                    if control is not None and edit_kind is not DocumentEditKind.DELETE
                    else 0
                ),
            )
            self._notify("root_identity_preflighted", intent)
            if raw:
                staged = self.revision_store.stage_blob(tenant, owner, document_id, raw)
                if staged != content_digest:
                    raise DocumentControlIntegrityError("external revision blob digest changed while staging")
                self._notify("after_blob_fsynced", intent)
            durable = self.control_store.prepare_intent(intent)
            self._assert_external_request_matches(
                durable,
                change_kind=change_kind,
                old_path=old_path,
                new_path=new_path,
                actor_binding=actor,
                evidence_reference=evidence,
                evidence_digest=evidence_hash,
                edit_summary=summary,
                before_digest=str(change.before_raw_digest or ""),
                after_digest=str(change.after_raw_digest or ""),
                match_raw_digests=idempotency_key is None,
            )
            self._notify("intent_prepared", durable)
        return self._resume_existing(durable, recovered=True)

    @staticmethod
    def _assert_external_request_matches(
        intent: DocumentCommitIntent,
        *,
        change_kind: str,
        old_path: str,
        new_path: str,
        actor_binding: str,
        evidence_reference: str,
        evidence_digest: str,
        edit_summary: str,
        before_digest: str,
        after_digest: str,
        match_raw_digests: bool,
    ) -> None:
        expected_kind = DocumentEditKind(change_kind)
        expected_old = old_path if expected_kind is not DocumentEditKind.CREATE else ""
        expected_new = new_path or (old_path if expected_kind is DocumentEditKind.UPDATE else "")
        intent_before_digest = next(
            (
                state.raw_sha256
                for state in (effect.before for effect in intent.effects)
                if isinstance(state, PresentPath)
            ),
            "",
        )
        intent_after_digest = next(
            (
                state.raw_sha256
                for state in (effect.after for effect in reversed(intent.effects))
                if isinstance(state, PresentPath)
            ),
            "",
        )
        if (
            intent.edit_kind is not expected_kind
            or intent.old_relative_path != expected_old
            or intent.new_relative_path != expected_new
            or intent.actor_binding != actor_binding
            or intent.evidence_reference != evidence_reference
            or intent.evidence_digest != evidence_digest
            or intent.edit_summary != edit_summary
            or (match_raw_digests and intent_before_digest != before_digest)
            or (match_raw_digests and intent_after_digest != after_digest)
        ):
            raise DocumentControlIntegrityError("external change idempotency key is already bound to another request")

    def _external_effects(
        self,
        tenant: str,
        owner: str,
        document_id: str,
        *,
        change_kind: str,
        old_path: str,
        new_path: str,
        before_digest: str,
        control: DocumentControlRecord | None,
    ) -> tuple[tuple[DocumentPathEffect, ...] | None, DocumentEditKind, bytes]:
        if change_kind in {"create", "update"}:
            path = MemoryDocumentPathPolicy.normalize_relative_path(new_path or old_path)
            state = self.document_store.read_state(tenant, owner, path)
            if not isinstance(state, PresentPath):
                raise DocumentCommitConflict("scanner-confirmed external file is no longer present")
            raw = self.document_store.read_raw(tenant, owner, relative_path=path)
            if hashlib.sha256(raw).hexdigest() != state.raw_sha256:
                raise DocumentCommitConflict("external file changed during durable adoption")
            if control is None:
                return (DocumentPathEffect(path, ABSENT, state),), DocumentEditKind.CREATE, raw
            before = PresentPath(old_path, before_digest, control.size)
            return (DocumentPathEffect(path, before, state),), DocumentEditKind.UPDATE, raw
        if change_kind == "rename":
            if control is None:
                raise DocumentCommitConflict("external rename has no durable before state")
            old = MemoryDocumentPathPolicy.normalize_relative_path(old_path)
            new = MemoryDocumentPathPolicy.normalize_relative_path(new_path)
            after = self.document_store.read_state(tenant, owner, new)
            if not isinstance(after, PresentPath):
                raise DocumentCommitConflict("external rename target is no longer present")
            raw = self.document_store.read_raw(tenant, owner, relative_path=new)
            before = PresentPath(old, before_digest, control.size)
            return (
                (
                    DocumentPathEffect(old, before, ABSENT),
                    DocumentPathEffect(new, ABSENT, after),
                ),
                DocumentEditKind.RENAME,
                raw,
            )
        if change_kind == "delete":
            if control is None:
                return None, DocumentEditKind.DELETE, b""
            old = MemoryDocumentPathPolicy.normalize_relative_path(old_path)
            if self.document_store.read_state(tenant, owner, old) != ABSENT:
                raise DocumentCommitConflict("external delete path is no longer absent")
            raw = self.revision_store.read_revision_blob(
                tenant,
                owner,
                document_id,
                self.revision_store.latest_revision(tenant, owner, document_id),
            )
            if hashlib.sha256(raw).hexdigest() != before_digest:
                raise DocumentCommitConflict("retained revision does not match the external delete before state")
            before = PresentPath(old, before_digest, len(raw))
            return (DocumentPathEffect(old, before, ABSENT),), DocumentEditKind.DELETE, raw
        raise ValueError("unsupported external document change kind")

    def restore_revision(
        self,
        plan: DocumentEditPlan,
        *,
        revision: int,
        actor_binding: str,
        evidence_reference: str,
    ) -> DocumentCommitResult:
        """通过新的 CREATE/UPDATE CAS 提交恢复历史正文。"""

        if plan.after_bytes is not None:
            raise ValueError("revision restore supplies its own immutable after bytes")
        self.erasure_store.assert_mutation_allowed(
            plan.tenant_id,
            plan.owner_user_id,
            plan.document_id,
        )
        if plan.edit_kind not in {DocumentEditKind.CREATE, DocumentEditKind.UPDATE}:
            raise ValueError("revision restore must be a CREATE or UPDATE plan")
        control = self.control_store.load_control(
            plan.tenant_id,
            plan.owner_user_id,
            plan.document_id,
        )
        barrier = self.control_store.load_publication_barrier(
            plan.tenant_id,
            plan.owner_user_id,
            plan.document_id,
        )
        restore_deletion_generation = 0
        if control is not None and control.status == "deleted":
            if barrier is None or barrier.status is not DocumentDeletionStatus.SOFT_FORGOTTEN:
                raise DocumentCommitConflict("deleted document has no restorable publication barrier")
            restore_deletion_generation = barrier.deletion_generation
        record = self.revision_store.load_revision(
            plan.tenant_id,
            plan.owner_user_id,
            plan.document_id,
            revision,
        )
        if record is None:
            raise DocumentNotFoundError("document revision does not exist")
        raw = self.revision_store.read_revision_blob(
            plan.tenant_id,
            plan.owner_user_id,
            plan.document_id,
            revision,
        )
        relative_path = plan.relative_path
        if not relative_path:
            if isinstance(plan.expected_state, PresentPath):
                relative_path = plan.expected_state.relative_path
            else:
                relative_path = record.relative_path
        restored_plan = replace(plan, relative_path=relative_path, after_bytes=raw)
        return self._commit(
            restored_plan,
            actor_binding=actor_binding,
            evidence_reference=evidence_reference,
            restore_deletion_generation=restore_deletion_generation,
        )
