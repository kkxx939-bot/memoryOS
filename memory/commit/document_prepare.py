"""单文档提交的计划校验、Intent 准备和锁边界。"""

from __future__ import annotations

import fcntl
import hashlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import NoReturn

from infrastructure.store.filesystem.file_lock import open_private_lock
from infrastructure.store.memory.control_store import (
    DocumentCommitIntent,
    DocumentControlIntegrityError,
    DocumentControlRecord,
    DocumentDeletionStatus,
    DocumentIntentStatus,
    DocumentPathEffect,
    DocumentRootIdentityGuard,
)
from infrastructure.store.memory.layout import tenant_control_root
from memory.commit.document_commit_types import (
    DocumentCommitConflict,
    DocumentCommitResult,
    _DocumentCommitState,
    _PreparedPlan,
)
from memory.core.model import (
    ABSENT,
    AbsentPath,
    DocumentEditKind,
    DocumentEditPlan,
    PresentPath,
    RawPathState,
    UnsafePath,
)
from memory.core.structure.frontmatter import parse_front_matter
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.ports.document_store import (
    DocumentConflictError,
    DocumentUnsafeError,
)


class _DocumentCommitPreparation(_DocumentCommitState):
    def _prepare_locked(
        self,
        prepared: _PreparedPlan,
        *,
        tenant: str,
        owner: str,
        document_id: str,
        intent_id: str,
        idempotency_digest: str,
        actor: str,
        evidence: str,
        evidence_digest: str,
        summary: str,
        restore_deletion_generation: int,
    ) -> tuple[DocumentCommitResult | DocumentCommitIntent, bool]:
        """在共享删除身份锁期间封存所有包含正文的准备产物。"""

        self.erasure_store.assert_mutation_allowed(tenant, owner, document_id)
        existing = self.control_store.load_intent(tenant, owner, intent_id)
        if existing is not None:
            self._assert_retry_matches(
                existing,
                prepared,
                actor_binding=actor,
                evidence_reference=evidence,
                evidence_digest=evidence_digest,
                edit_summary=summary,
                restored_from_deletion_generation=restore_deletion_generation,
            )
            return existing, True
        active = tuple(
            intent
            for intent in self.control_store.incomplete_intents(tenant, owner)
            if intent.document_id == document_id and intent.intent_id != intent_id
        )
        if active:
            raise DocumentConflictError("another durable intent for this document must complete recovery first")

        self._require_prepared_live_vector(prepared, expected="before")
        with self.control_store.root_identity_lock(tenant, owner) as root_guard:
            return self._prepare_new_intent_under_owner_lock(
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
                root_guard=root_guard,
            )

    def _prepare_new_intent_under_owner_lock(
        self,
        prepared: _PreparedPlan,
        *,
        tenant: str,
        owner: str,
        document_id: str,
        intent_id: str,
        idempotency_digest: str,
        actor: str,
        evidence: str,
        evidence_digest: str,
        summary: str,
        restore_deletion_generation: int,
        root_guard: DocumentRootIdentityGuard,
    ) -> tuple[DocumentCommitResult | DocumentCommitIntent, bool]:
        """在同一所有者锁内通过 PREPARED 状态发布初始授权。"""

        self._preflight_new_intent_root(
            prepared.tenant_id,
            prepared.owner_user_id,
            edit_kind=prepared.edit_kind,
            allow_initial_publication=prepared.edit_kind is DocumentEditKind.CREATE,
            root_guard=root_guard,
        )
        prepared = self._load_revision_bytes(prepared)
        control, restored_from_deletion_generation = self._validate_control_boundary(
            prepared,
            restore_deletion_generation=restore_deletion_generation,
        )
        if self._is_no_op(prepared):
            return (
                DocumentCommitResult(
                    intent_id=intent_id,
                    status=DocumentIntentStatus.COMPLETED,
                    event=None,
                    control=control,
                    revision=None,
                    no_op=True,
                ),
                False,
            )

        latest_revision = self.revision_store.latest_revision(tenant, owner, document_id)
        latest_record = (
            self.revision_store.load_revision(tenant, owner, document_id, latest_revision) if latest_revision else None
        )
        current_revision = control.logical_revision if control is not None else 0
        current_projection = control.projection_generation if control is not None else 0
        logical_revision = max(latest_revision, current_revision) + 1
        historical_projection = latest_record.projection_generation if latest_record is not None else 0
        projection_generation = max(historical_projection, current_projection) + 1
        now = self.clock()
        event_suffix = hashlib.sha256(f"{intent_id}:event".encode()).hexdigest()
        job_suffix = hashlib.sha256(f"{intent_id}:projection".encode()).hexdigest()
        intent = DocumentCommitIntent(
            intent_id=intent_id,
            idempotency_digest=idempotency_digest,
            identity_digest=None,
            tenant_id=tenant,
            owner_user_id=owner,
            document_id=document_id,
            edit_kind=prepared.edit_kind,
            effects=prepared.effects,
            after_blob_digest=prepared.after_blob_digest,
            revision_blob_digest=prepared.revision_blob_digest,
            revision_blob_role=prepared.revision_blob_role,
            logical_revision=logical_revision,
            projection_generation=projection_generation,
            event_id=f"memchg_{event_suffix}",
            projection_job_id=f"memory_projection_{job_suffix}",
            old_relative_path=prepared.old_relative_path,
            new_relative_path=prepared.new_relative_path,
            actor_binding=actor,
            evidence_reference=evidence,
            evidence_digest=evidence_digest,
            edit_summary=summary,
            status=DocumentIntentStatus.PREPARED,
            created_at=now,
            updated_at=now,
            restored_from_deletion_generation=restored_from_deletion_generation,
        )
        self._notify("root_identity_preflighted", intent)
        if prepared.revision_blob_digest:
            staged = self.revision_store.stage_blob(tenant, owner, document_id, prepared.revision_bytes)
            if staged != prepared.revision_blob_digest:
                raise DocumentControlIntegrityError("staged revision blob digest changed during preparation")
            self._notify("after_blob_fsynced", intent)
        durable = self.control_store.prepare_intent(intent)
        if durable.identity_digest != intent.identity_digest:
            self._assert_retry_matches(
                durable,
                prepared,
                actor_binding=actor,
                evidence_reference=evidence,
                evidence_digest=evidence_digest,
                edit_summary=summary,
                restored_from_deletion_generation=restored_from_deletion_generation,
            )
            if durable.edit_kind is DocumentEditKind.DELETE:
                self.control_store.ensure_soft_forget_barrier(durable)
            return durable, True
        if durable.edit_kind is DocumentEditKind.DELETE:
            self.control_store.ensure_soft_forget_barrier(durable)
        self._notify("intent_prepared", durable)
        return durable, False

    def _prepare_plan(
        self,
        plan: DocumentEditPlan,
        *,
        tenant: str,
        owner: str,
        document_id: str,
    ) -> _PreparedPlan:
        del tenant, owner
        if isinstance(plan.expected_state, UnsafePath) or isinstance(plan.expected_new_state, UnsafePath):
            raise DocumentUnsafeError("UNSAFE raw state can never authorize a document commit")
        edit_kind = DocumentEditKind(plan.edit_kind)
        if edit_kind == DocumentEditKind.CREATE:
            if not isinstance(plan.expected_state, AbsentPath) or plan.after_bytes is None:
                raise ValueError("CREATE requires before=ABSENT and exact after bytes")
            relative = MemoryDocumentPathPolicy.normalize_relative_path(plan.relative_path)
            after = bytes(plan.after_bytes)
            self._validate_after_bytes(document_id, after)
            digest = hashlib.sha256(after).hexdigest()
            after_state = PresentPath(relative, digest, len(after))
            return _PreparedPlan(
                tenant_id=plan.tenant_id,
                owner_user_id=plan.owner_user_id,
                document_id=document_id,
                edit_kind=edit_kind,
                effects=(DocumentPathEffect(relative, ABSENT, after_state),),
                after_bytes=after,
                revision_bytes=after,
                after_blob_digest=digest,
                revision_blob_digest=digest,
                revision_blob_role="after",
                old_relative_path="",
                new_relative_path=relative,
            )
        if not isinstance(plan.expected_state, PresentPath):
            raise ValueError(f"{edit_kind.value.upper()} requires an exact PRESENT before state")
        relative = MemoryDocumentPathPolicy.normalize_relative_path(
            plan.relative_path or plan.expected_state.relative_path
        )
        if plan.expected_state.relative_path != relative:
            raise ValueError("plan path is detached from its expected PRESENT state")

        if edit_kind == DocumentEditKind.UPDATE:
            if plan.after_bytes is None or plan.new_relative_path:
                raise ValueError("UPDATE requires exact after bytes and cannot rename")
            after = bytes(plan.after_bytes)
            self._validate_after_bytes(document_id, after)
            digest = hashlib.sha256(after).hexdigest()
            after_state = PresentPath(relative, digest, len(after))
            return _PreparedPlan(
                tenant_id=plan.tenant_id,
                owner_user_id=plan.owner_user_id,
                document_id=document_id,
                edit_kind=edit_kind,
                effects=(DocumentPathEffect(relative, plan.expected_state, after_state),),
                after_bytes=after,
                revision_bytes=after,
                after_blob_digest=digest,
                revision_blob_digest=digest,
                revision_blob_role="after",
                old_relative_path=relative,
                new_relative_path=relative,
            )
        if edit_kind == DocumentEditKind.DELETE:
            if plan.after_bytes is not None:
                raise ValueError("DELETE cannot use fictitious after bytes")
            if plan.new_relative_path:
                raise ValueError("DELETE cannot have a new path")
            return _PreparedPlan(
                tenant_id=plan.tenant_id,
                owner_user_id=plan.owner_user_id,
                document_id=document_id,
                edit_kind=edit_kind,
                effects=(DocumentPathEffect(relative, plan.expected_state, ABSENT),),
                after_bytes=None,
                revision_bytes=b"",
                after_blob_digest="",
                revision_blob_digest=plan.expected_state.raw_sha256,
                revision_blob_role="before_delete",
                old_relative_path=relative,
                new_relative_path="",
            )
        if edit_kind != DocumentEditKind.RENAME:
            raise ValueError("unsupported document edit kind")
        new_relative = MemoryDocumentPathPolicy.normalize_relative_path(plan.new_relative_path)
        if new_relative == relative or not isinstance(plan.expected_new_state, AbsentPath):
            raise ValueError("RENAME requires a distinct target with expected ABSENT state")
        if plan.after_bytes is None:
            rename_after: bytes | None = None
            revision_bytes = b""
            after_digest = plan.expected_state.raw_sha256
            after_size = plan.expected_state.size
        else:
            rename_after = bytes(plan.after_bytes)
            self._validate_after_bytes(document_id, rename_after)
            revision_bytes = rename_after
            after_digest = hashlib.sha256(rename_after).hexdigest()
            after_size = len(rename_after)
        return _PreparedPlan(
            tenant_id=plan.tenant_id,
            owner_user_id=plan.owner_user_id,
            document_id=document_id,
            edit_kind=edit_kind,
            effects=(
                DocumentPathEffect(relative, plan.expected_state, ABSENT),
                DocumentPathEffect(
                    new_relative,
                    ABSENT,
                    PresentPath(new_relative, after_digest, after_size),
                ),
            ),
            after_bytes=rename_after,
            revision_bytes=revision_bytes,
            after_blob_digest=after_digest,
            revision_blob_digest=after_digest,
            revision_blob_role="after",
            old_relative_path=relative,
            new_relative_path=new_relative,
        )

    def _load_revision_bytes(self, prepared: _PreparedPlan) -> _PreparedPlan:
        if prepared.revision_bytes:
            return prepared
        if prepared.edit_kind not in {DocumentEditKind.DELETE, DocumentEditKind.RENAME}:
            return prepared
        raw = self.document_store.read_raw(
            prepared.tenant_id,
            prepared.owner_user_id,
            relative_path=prepared.old_relative_path,
        )
        if hashlib.sha256(raw).hexdigest() != prepared.revision_blob_digest:
            raise DocumentConflictError("live document bytes differ from the prepared revision digest")
        before = prepared.effects[0].before
        if not isinstance(before, PresentPath) or len(raw) != before.size:
            raise DocumentConflictError("live document bytes differ from the prepared revision size")
        return replace(prepared, revision_bytes=raw)

    def _validate_after_bytes(self, document_id: str, after: bytes) -> None:
        max_file_bytes = int(getattr(self.document_store, "max_file_bytes", 2 * 1024 * 1024))
        if len(after) > max_file_bytes:
            raise DocumentUnsafeError("document after bytes exceed the configured byte limit")
        parsed = parse_front_matter(
            after,
            max_header_bytes=int(getattr(self.document_store, "max_front_matter_bytes", 32 * 1024)),
            max_depth=int(getattr(self.document_store, "max_front_matter_depth", 12)),
        )
        if parsed.document_id != document_id:
            raise DocumentConflictError("document after bytes cannot change document_id")

    def _validate_control_boundary(
        self,
        prepared: _PreparedPlan,
        *,
        restore_deletion_generation: int,
    ) -> tuple[DocumentControlRecord | None, int]:
        record = self.control_store.load_control(
            prepared.tenant_id,
            prepared.owner_user_id,
            prepared.document_id,
        )
        barrier = self.control_store.load_publication_barrier(
            prepared.tenant_id,
            prepared.owner_user_id,
            prepared.document_id,
        )
        if barrier is not None and barrier.status is DocumentDeletionStatus.HARD_ERASED:
            raise DocumentCommitConflict("hard-erased document identity cannot be reused")
        if record is None:
            if barrier is not None:
                raise DocumentCommitConflict(
                    "deleted document identity requires retained control and an explicit restore"
                )
            if restore_deletion_generation:
                raise DocumentCommitConflict("explicit restore is detached from a deletion barrier")
            return None, 0
        before_present = next(
            (effect.before for effect in prepared.effects if isinstance(effect.before, PresentPath)),
            None,
        )
        if isinstance(before_present, PresentPath):
            if (
                record.status != "present"
                or record.relative_path != before_present.relative_path
                or record.raw_sha256 != before_present.raw_sha256
                or record.size != before_present.size
            ):
                raise DocumentCommitConflict(
                    "control metadata differs from live expected state; rescan is required and live Markdown was preserved"
                )
        elif record.status == "present":
            raise DocumentCommitConflict("document identity is already controlled at a live path")
        if record.status == "deleted":
            if (
                prepared.edit_kind is not DocumentEditKind.CREATE
                or barrier is None
                or barrier.status is not DocumentDeletionStatus.SOFT_FORGOTTEN
                or restore_deletion_generation != barrier.deletion_generation
                or record.projection_generation != barrier.deletion_generation
            ):
                raise DocumentCommitConflict("soft-forgotten document identity requires an explicit revision restore")
            return record, barrier.deletion_generation
        if restore_deletion_generation:
            raise DocumentCommitConflict("explicit restore marker is not bound to a deleted document")
        inherited_restore = record.restored_from_deletion_generation
        if barrier is not None and barrier.status is DocumentDeletionStatus.SOFT_FORGOTTEN:
            if (
                prepared.edit_kind is DocumentEditKind.DELETE
                and record.projection_generation > barrier.deletion_generation
                and inherited_restore == barrier.deletion_generation
            ):
                return record, 0
            if (
                inherited_restore != barrier.deletion_generation
                or record.projection_generation <= barrier.deletion_generation
            ):
                raise DocumentControlIntegrityError("live control is detached from its retained soft-forget barrier")
        elif inherited_restore:
            raise DocumentControlIntegrityError(
                "live control claims restored lineage without a durable deletion barrier"
            )
        return record, inherited_restore

    @staticmethod
    def _is_no_op(prepared: _PreparedPlan) -> bool:
        if prepared.edit_kind != DocumentEditKind.UPDATE:
            return False
        effect = prepared.effects[0]
        return effect.before == effect.after

    def _assert_retry_matches(
        self,
        intent: DocumentCommitIntent,
        prepared: _PreparedPlan,
        *,
        actor_binding: str,
        evidence_reference: str,
        evidence_digest: str,
        edit_summary: str,
        restored_from_deletion_generation: int,
    ) -> None:
        expected = (
            prepared.edit_kind,
            prepared.effects,
            prepared.after_blob_digest,
            prepared.revision_blob_digest,
            prepared.revision_blob_role,
            prepared.old_relative_path,
            prepared.new_relative_path,
            actor_binding,
            evidence_reference,
            evidence_digest,
            edit_summary,
            restored_from_deletion_generation,
        )
        actual = (
            intent.edit_kind,
            intent.effects,
            intent.after_blob_digest,
            intent.revision_blob_digest,
            intent.revision_blob_role,
            intent.old_relative_path,
            intent.new_relative_path,
            intent.actor_binding,
            intent.evidence_reference,
            intent.evidence_digest,
            intent.edit_summary,
            intent.restored_from_deletion_generation,
        )
        if actual != expected:
            raise DocumentControlIntegrityError("idempotency key is already bound to another document edit")

    def _read_after_blob(self, intent: DocumentCommitIntent) -> bytes:
        if not intent.after_blob_digest:
            raise DocumentControlIntegrityError("document install requires an immutable after blob")
        return self.revision_store.read_blob(
            intent.tenant_id,
            intent.owner_user_id,
            intent.document_id,
            intent.after_blob_digest,
        )

    def _capture_intent_live(self, intent: DocumentCommitIntent) -> tuple[RawPathState, ...]:
        return tuple(
            self.document_store.read_state(intent.tenant_id, intent.owner_user_id, effect.relative_path)
            for effect in intent.effects
        )

    def _require_prepared_live_vector(self, prepared: _PreparedPlan, *, expected: str) -> None:
        live = tuple(
            self.document_store.read_state(prepared.tenant_id, prepared.owner_user_id, effect.relative_path)
            for effect in prepared.effects
        )
        wanted = tuple(getattr(effect, expected) for effect in prepared.effects)
        if any(isinstance(state, UnsafePath) for state in live):
            raise DocumentUnsafeError("live document vector contains an UNSAFE state")
        if live != wanted:
            raise DocumentConflictError(f"live document vector does not match exact {expected} state")

    @contextmanager
    def _document_lock(self, intent: DocumentCommitIntent) -> Iterator[None]:
        with self._document_identity_lock(
            intent.tenant_id,
            intent.owner_user_id,
            intent.document_id,
        ):
            yield

    @contextmanager
    def _document_identity_lock(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> Iterator[None]:
        artifact_root = tenant_control_root(self.control_store.root, tenant_id)
        lock_path = artifact_root / "system" / "memory-documents" / owner_user_id / "locks" / f"{document_id}.lock"
        descriptor = open_private_lock(lock_path, root=artifact_root)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            if self.path_lock is None:
                yield
                return
            lock_digest = hashlib.sha256(f"{tenant_id}\0{owner_user_id}\0{document_id}".encode()).hexdigest()
            with self.path_lock.acquire(f"memory-document:{lock_digest}") as guard:
                with guard.fenced():
                    yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _mark_conflicted(self, intent: DocumentCommitIntent, reason: str) -> NoReturn:
        self.control_store.update_intent(
            intent,
            DocumentIntentStatus.CONFLICTED,
            updated_at=self.clock(),
            conflict_reason=reason,
        )
        raise DocumentCommitConflict(reason)
