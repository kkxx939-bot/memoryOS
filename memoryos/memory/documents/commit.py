"""Roll-forward-only commit and recovery for live Markdown memory documents."""

from __future__ import annotations

import fcntl
import hashlib
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NoReturn

from memoryos.contextdb.store.queue_store import QueueJob, QueueStore
from memoryos.contextdb.transaction.path_lock import PathLock
from memoryos.core.clock import utc_now
from memoryos.core.file_lock import open_private_lock
from memoryos.memory.documents.control_store import (
    DocumentCommitIntent,
    DocumentControlIntegrityError,
    DocumentControlRecord,
    DocumentDeletionStatus,
    DocumentIntentStatus,
    DocumentPathEffect,
    DocumentRootIdentityGuard,
    MemoryDocumentControlStore,
    document_intent_id,
)
from memoryos.memory.documents.erase import MemoryDocumentEraseStore
from memoryos.memory.documents.frontmatter import matches_adopted_source, parse_front_matter, validate_document_id
from memoryos.memory.documents.layout import tenant_control_root
from memoryos.memory.documents.model import (
    ABSENT,
    AbsentPath,
    DocumentChangeEvent,
    DocumentEditKind,
    DocumentEditPlan,
    ManagedDocument,
    PresentPath,
    RawPathState,
    ScanGeneration,
    UnsafePath,
)
from memoryos.memory.documents.path_policy import MemoryDocumentPathPolicy
from memoryos.memory.documents.revision_store import (
    DocumentRevisionRecord,
    MemoryDocumentRevisionStore,
)
from memoryos.memory.documents.store import (
    DocumentConflictError,
    DocumentNotFoundError,
    DocumentUnsafeError,
    MemoryDocumentStore,
)

if TYPE_CHECKING:
    from memoryos.memory.documents.scanner import ExternalDocumentChange


class DocumentCommitConflict(DocumentConflictError):
    """Live Markdown or durable identity is a third state and was preserved."""


@dataclass(frozen=True)
class DocumentCommitResult:
    intent_id: str
    status: DocumentIntentStatus
    event: DocumentChangeEvent | None
    control: DocumentControlRecord | None
    revision: DocumentRevisionRecord | None
    no_op: bool = False
    recovered: bool = False


@dataclass(frozen=True)
class DocumentRecoveryReport:
    completed: tuple[DocumentCommitResult, ...]
    conflicted_intent_ids: tuple[str, ...]


@dataclass(frozen=True)
class _PreparedPlan:
    tenant_id: str
    owner_user_id: str
    document_id: str
    edit_kind: DocumentEditKind
    effects: tuple[DocumentPathEffect, ...]
    after_bytes: bytes | None
    revision_bytes: bytes
    after_blob_digest: str
    revision_blob_digest: str
    revision_blob_role: str
    old_relative_path: str
    new_relative_path: str


FaultHook = Callable[[str, DocumentCommitIntent], None]


class MemoryDocumentCommitter:
    """Independent document committer; never delegates Markdown to OperationCommitter."""

    def __init__(
        self,
        document_store: MemoryDocumentStore,
        control_store: MemoryDocumentControlStore,
        revision_store: MemoryDocumentRevisionStore,
        projection_queue: QueueStore,
        *,
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
        self.erasure_store = MemoryDocumentEraseStore(control_store.root)

    def commit(
        self,
        plan: DocumentEditPlan,
        *,
        actor_binding: str,
        evidence_reference: str,
    ) -> DocumentCommitResult:
        """Prepare and roll forward one exact document CAS."""

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
        """Bind an adoption root before its unmanaged source CAS."""

        tenant = MemoryDocumentPathPolicy.trusted_segment(tenant_id, "tenant_id")
        owner = MemoryDocumentPathPolicy.trusted_segment(owner_user_id, "owner_user_id")
        identifier = validate_document_id(document_id)
        with self._document_identity_lock(
            tenant,
            owner,
            identifier,
        ), self.control_store.root_identity_lock(tenant, owner) as root_guard:
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
        """Verify a previously preflighted adoption retry without backfill."""

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
        """Internal commit entry carrying a non-forgeable explicit-restore marker."""

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
        """Seal all body-bearing preparation while sharing the erasure identity lock."""

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
        """Publish initial authority through PREPARED under one owner lock."""

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
        # New blob publication and PREPARED intent creation hold this same
        # owner lock, so GC can never race the small blob-before-intent window.
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
        """Journal one scanner-confirmed external edit without rewriting its bytes.

        The prepared intent's exact after vector already exists in the live
        filesystem.  Normal recovery therefore observes ``live == after`` and
        advances only the content-free event, revision, control, and queue
        tail.  A later third state is preserved by the usual recovery guard.
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
                    raise DocumentCommitConflict(
                        "scanner CREATE is detached from its durable adoption receipt path"
                    )
                raw = self.document_store.read_raw(
                    tenant,
                    owner,
                    document_id=document_id,
                )
                if (
                    hashlib.sha256(raw).hexdigest() != str(change.after_raw_digest or "")
                    or not matches_adopted_source(
                        raw,
                        receipt.document_id,
                        receipt.expected_raw_sha256,
                    )
                ):
                    raise DocumentCommitConflict(
                        "scanner CREATE does not match the exact durable adoption receipt"
                    )
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
            _bounded_text(idempotency_key, "idempotency_key", 512)
            if idempotency_key is not None
            else default_identity
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
        with self._document_identity_lock(
            tenant,
            owner,
            document_id,
        ), self.control_store.root_identity_lock(tenant, owner) as root_guard:
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
                if (
                    control.raw_sha256 == str(change.after_raw_digest or "")
                    and control.relative_path == (new_path or old_path)
                ):
                    self._verify_existing_root_identity(tenant, owner)
                    return None
                if (
                    control.raw_sha256 != str(change.before_raw_digest or "")
                    or control.relative_path != old_path
                ):
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
                allow_initial_publication=(
                    edit_kind is DocumentEditKind.CREATE and adoption_receipt is None
                ),
                root_guard=root_guard,
            )
            latest_revision = self.revision_store.latest_revision(tenant, owner, document_id)
            latest_record = (
                self.revision_store.load_revision(tenant, owner, document_id, latest_revision)
                if latest_revision
                else None
            )
            logical_revision = max(latest_revision, control.logical_revision if control is not None else 0) + 1
            projection_generation = max(
                latest_record.projection_generation if latest_record is not None else 0,
                control.projection_generation if control is not None else 0,
            ) + 1
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
            raise DocumentControlIntegrityError(
                "external change idempotency key is already bound to another request"
            )

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
                DocumentPathEffect(old, before, ABSENT),
                DocumentPathEffect(new, ABSENT, after),
            ), DocumentEditKind.RENAME, raw
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
        """Restore historical bytes through a new CREATE/UPDATE CAS commit."""

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
                raise DocumentCommitConflict(
                    "soft-forgotten document identity requires an explicit revision restore"
                )
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
                raise DocumentControlIntegrityError(
                    "live control is detached from its retained soft-forget barrier"
                )
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

    def _resume_existing(self, intent: DocumentCommitIntent, *, recovered: bool) -> DocumentCommitResult:
        try:
            return self._resume_existing_once(intent, recovered=recovered)
        except BaseException as exc:
            self._enqueue_retryable_recovery(
                intent.tenant_id,
                intent.owner_user_id,
                intent.document_id,
                intent.intent_id,
                exc,
            )
            raise

    def _resume_existing_once(self, intent: DocumentCommitIntent, *, recovered: bool) -> DocumentCommitResult:
        if intent.status == DocumentIntentStatus.CONFLICTED:
            raise DocumentCommitConflict(intent.conflict_reason or "document intent is conflicted")
        if intent.status == DocumentIntentStatus.COMPLETED:
            return self._completed_result(intent, recovered=recovered)
        with self._document_lock(intent):
            self._notify("lock_acquired", intent)
            self.erasure_store.assert_mutation_allowed(
                intent.tenant_id,
                intent.owner_user_id,
                intent.document_id,
            )
            current = self.control_store.load_intent(intent.tenant_id, intent.owner_user_id, intent.intent_id)
            if current is None or current.identity_digest != intent.identity_digest:
                raise DocumentControlIntegrityError("document intent disappeared or changed while locked")
            if current.status == DocumentIntentStatus.COMPLETED:
                return self._completed_result(current, recovered=recovered)
            if current.status == DocumentIntentStatus.CONFLICTED:
                raise DocumentCommitConflict(current.conflict_reason or "document intent is conflicted")
            cleanup_temps = getattr(self.document_store, "cleanup_operation_temps", None)
            if callable(cleanup_temps):
                expected_temp_digests: dict[str, str] = {}
                if current.edit_kind in {DocumentEditKind.CREATE, DocumentEditKind.UPDATE}:
                    effect = current.effects[0]
                    if isinstance(effect.after, PresentPath):
                        expected_temp_digests[effect.relative_path] = effect.after.raw_sha256
                elif current.edit_kind is DocumentEditKind.RENAME:
                    old_effect, new_effect = current.effects
                    if (
                        isinstance(old_effect.before, PresentPath)
                        and isinstance(new_effect.after, PresentPath)
                        and (
                            old_effect.before.raw_sha256 != new_effect.after.raw_sha256
                            or old_effect.before.size != new_effect.after.size
                        )
                    ):
                        expected_temp_digests[new_effect.relative_path] = new_effect.after.raw_sha256
                if expected_temp_digests:
                    cleanup_temps(
                        current.tenant_id,
                        current.owner_user_id,
                        expected_temp_digests,
                        current.intent_id,
                    )
            self._verify_existing_root_identity(
                current.tenant_id,
                current.owner_user_id,
            )
            if current.edit_kind is DocumentEditKind.DELETE:
                self.control_store.ensure_soft_forget_barrier(current)
            live = self._capture_intent_live(current)
            self._notify("live_state_reread", current)
            before = tuple(effect.before for effect in current.effects)
            after = tuple(effect.after for effect in current.effects)
            if any(isinstance(state, UnsafePath) for state in live):
                self._mark_conflicted(current, "live document vector contains an UNSAFE path state")
            if live == before:
                self._install(current)
                self._verify_existing_root_identity(
                    current.tenant_id,
                    current.owner_user_id,
                )
                self._notify("after_installed", current)
                live = self._capture_intent_live(current)
            if live != after:
                self._mark_conflicted(
                    current,
                    "live document vector is neither the exact before state nor the exact after state",
                )
            self._verify_existing_root_identity(
                current.tenant_id,
                current.owner_user_id,
            )
            current = self.control_store.update_intent(
                current,
                DocumentIntentStatus.INSTALLED,
                updated_at=self.clock(),
            )
            return self._complete_tail(current, recovered=recovered)

    def _install(self, intent: DocumentCommitIntent) -> None:
        if intent.edit_kind != DocumentEditKind.CREATE:
            self._require_registered_before(intent)
        try:
            if intent.edit_kind == DocumentEditKind.CREATE:
                effect = intent.effects[0]
                raw = self._read_after_blob(intent)
                self.document_store.create(
                    intent.tenant_id,
                    intent.owner_user_id,
                    effect.relative_path,
                    raw,
                    expected=effect.before,
                    operation_id=intent.intent_id,
                    fault_hook=lambda stage: self._notify(stage, intent),
                )
            elif intent.edit_kind == DocumentEditKind.UPDATE:
                effect = intent.effects[0]
                self.document_store.replace(
                    intent.tenant_id,
                    intent.owner_user_id,
                    intent.document_id,
                    self._read_after_blob(intent),
                    expected_state=effect.before,
                    operation_id=intent.intent_id,
                    fault_hook=lambda stage: self._notify(stage, intent),
                )
            elif intent.edit_kind == DocumentEditKind.DELETE:
                effect = intent.effects[0]
                self.document_store.delete(
                    intent.tenant_id,
                    intent.owner_user_id,
                    intent.document_id,
                    expected_state=effect.before,
                    operation_id=intent.intent_id,
                    fault_hook=lambda stage: self._notify(stage, intent),
                )
            elif intent.edit_kind == DocumentEditKind.RENAME:
                old_effect, new_effect = intent.effects
                content_changed = bool(
                    isinstance(old_effect.before, PresentPath)
                    and isinstance(new_effect.after, PresentPath)
                    and (
                        old_effect.before.raw_sha256 != new_effect.after.raw_sha256
                        or old_effect.before.size != new_effect.after.size
                    )
                )
                self.document_store.rename(
                    intent.tenant_id,
                    intent.owner_user_id,
                    intent.document_id,
                    new_effect.relative_path,
                    expected_old=old_effect.before,
                    expected_new=new_effect.before,
                    after_bytes=self._read_after_blob(intent) if content_changed else None,
                    operation_id=intent.intent_id,
                    fault_hook=lambda stage: self._notify(stage, intent),
                )
            else:  # pragma: no cover - enum is closed.
                raise ValueError("unsupported document edit kind")
        except (DocumentConflictError, DocumentNotFoundError, DocumentUnsafeError):
            live = self._capture_intent_live(intent)
            after = tuple(effect.after for effect in intent.effects)
            before = tuple(effect.before for effect in intent.effects)
            if live == after:
                return
            if live != before:
                self._mark_conflicted(intent, "document store observed a third state during atomic install")
            raise

    def _require_registered_before(self, intent: DocumentCommitIntent) -> None:
        scan = self.document_store.full_scan(intent.tenant_id, intent.owner_user_id)
        if not scan.complete or scan.errors:
            self._mark_conflicted(intent, "document registration scan is incomplete")
        old_path = intent.old_relative_path
        before = next(
            (
                effect.before
                for effect in intent.effects
                if effect.relative_path == old_path and isinstance(effect.before, PresentPath)
            ),
            None,
        )
        match = next(
            (
                item
                for item in scan.registrations
                if isinstance(item, ManagedDocument)
                and item.document_id == intent.document_id
                and item.relative_path == old_path
            ),
            None,
        )
        if not isinstance(before, PresentPath) or match is None:
            self._mark_conflicted(intent, "document registration is absent, duplicate or detached from its path")
        if match.raw_sha256 != before.raw_sha256 or match.size != before.size:
            self._mark_conflicted(intent, "document registration no longer matches the expected raw state")

    def _complete_tail(self, intent: DocumentCommitIntent, *, recovered: bool) -> DocumentCommitResult:
        event = self._event(intent)
        self.control_store.append_event(intent, event)
        self._notify("event_appended", intent)
        intent = self.control_store.update_intent(
            intent,
            DocumentIntentStatus.EVENT_APPENDED,
            updated_at=self.clock(),
        )
        revision = self.revision_store.record_revision(intent, event)
        self._notify("revision_recorded", intent)
        control = self._control_record(intent, event)
        self.control_store.write_control(control)
        self._notify("control_recorded", intent)
        self.projection_queue.enqueue(self._projection_job(intent, event))
        self._notify("projection_enqueued", intent)
        intent = self.control_store.update_intent(
            intent,
            DocumentIntentStatus.PROJECTION_ENQUEUED,
            updated_at=self.clock(),
        )
        intent = self.control_store.update_intent(
            intent,
            DocumentIntentStatus.COMPLETED,
            updated_at=self.clock(),
        )
        self._notify("completed", intent)
        return DocumentCommitResult(
            intent_id=intent.intent_id,
            status=intent.status,
            event=event,
            control=control,
            revision=revision,
            recovered=recovered,
        )

    def _completed_result(self, intent: DocumentCommitIntent, *, recovered: bool) -> DocumentCommitResult:
        self._verify_existing_root_identity(
            intent.tenant_id,
            intent.owner_user_id,
        )
        event = self.control_store.load_event(intent)
        if event is None:
            raise DocumentControlIntegrityError("completed document intent has no durable change event")
        control = self.control_store.load_control(intent.tenant_id, intent.owner_user_id, intent.document_id)
        revision = self.revision_store.load_revision(
            intent.tenant_id,
            intent.owner_user_id,
            intent.document_id,
            intent.logical_revision,
        )
        if control is None or revision is None:
            raise DocumentControlIntegrityError("completed document intent has incomplete durable metadata")
        return DocumentCommitResult(
            intent_id=intent.intent_id,
            status=intent.status,
            event=event,
            control=control,
            revision=revision,
            recovered=recovered,
        )

    def _preflight_new_intent_root(
        self,
        tenant_id: str,
        owner_user_id: str,
        *,
        edit_kind: DocumentEditKind,
        allow_initial_publication: bool,
        root_guard: DocumentRootIdentityGuard,
    ) -> None:
        """Bind a new CREATE before durable intent/blob/source mutation."""

        if allow_initial_publication and edit_kind is not DocumentEditKind.CREATE:
            raise DocumentControlIntegrityError(
                "only a new CREATE may publish an owner's initial source root identity"
            )
        if edit_kind is DocumentEditKind.CREATE:
            probe = getattr(self.document_store, "probe_write_capabilities", None)
            if not callable(probe):
                raise DocumentControlIntegrityError(
                    "document store cannot establish a real owner root before CREATE"
                )
            probe(tenant_id, owner_user_id)

        scan = self.document_store.full_scan(tenant_id, owner_user_id)
        self._require_publishable_root_scan(scan)
        durable = self.control_store.load_root_identity(tenant_id, owner_user_id)
        if durable is None:
            if not allow_initial_publication:
                raise DocumentControlIntegrityError(
                    "existing document authority is missing its durable source root identity"
                )
            if self.control_store.controls(tenant_id, owner_user_id):
                raise DocumentControlIntegrityError(
                    "existing document controls are missing their durable source root identity"
                )
            if self.control_store.incomplete_intents(tenant_id, owner_user_id):
                raise DocumentControlIntegrityError(
                    "existing document intents are missing their durable source root identity"
                )
            root_guard.ensure(
                scan.root_identity,
            )
            return
        if durable.root_identity != scan.root_identity:
            raise DocumentControlIntegrityError(
                "document source root identity changed and requires explicit reset"
            )

    def _verify_existing_root_identity(
        self,
        tenant_id: str,
        owner_user_id: str,
    ) -> None:
        """Verify a durable-intent root without ever creating its authority."""

        scan = self.document_store.full_scan(tenant_id, owner_user_id)
        self._require_publishable_root_scan(scan)
        durable = self.control_store.load_root_identity(tenant_id, owner_user_id)
        if durable is None:
            raise DocumentControlIntegrityError(
                "durable document intent is missing its source root identity"
            )
        if durable.root_identity != scan.root_identity:
            raise DocumentControlIntegrityError(
                "document source root identity changed and requires explicit reset"
            )

    @staticmethod
    def _require_publishable_root_scan(scan: ScanGeneration) -> None:
        if (
            not scan.complete
            or scan.errors
            or len(scan.root_identity) != 32
            or any(character not in "0123456789abcdef" for character in scan.root_identity)
        ):
            raise DocumentControlIntegrityError(
                "document commit cannot bind an incomplete or unsafe source root scan"
            )

    def _event(self, intent: DocumentCommitIntent) -> DocumentChangeEvent:
        before_digest = next(
            (
                state.raw_sha256
                for state in (effect.before for effect in intent.effects)
                if isinstance(state, PresentPath)
            ),
            "",
        )
        after_digest = next(
            (
                state.raw_sha256
                for state in (effect.after for effect in reversed(intent.effects))
                if isinstance(state, PresentPath)
            ),
            "",
        )
        return DocumentChangeEvent(
            event_id=intent.event_id,
            tenant_id=intent.tenant_id,
            owner_user_id=intent.owner_user_id,
            document_id=intent.document_id,
            edit_kind=intent.edit_kind,
            old_relative_path=intent.old_relative_path,
            new_relative_path=intent.new_relative_path,
            before_raw_digest=before_digest,
            after_raw_digest=after_digest,
            logical_revision=intent.logical_revision,
            projection_generation=intent.projection_generation,
            occurred_at=intent.created_at,
            actor_binding=intent.actor_binding,
            evidence_reference=intent.evidence_reference,
            evidence_digest=intent.evidence_digest,
            edit_summary=intent.edit_summary,
        )

    @staticmethod
    def _control_record(intent: DocumentCommitIntent, event: DocumentChangeEvent) -> DocumentControlRecord:
        present = next(
            (
                (effect.relative_path, effect.after)
                for effect in reversed(intent.effects)
                if isinstance(effect.after, PresentPath)
            ),
            None,
        )
        if present is None:
            relative_path = intent.old_relative_path
            raw_sha256 = ""
            size = 0
            status = "deleted"
            restored_from_deletion_generation = 0
        else:
            relative_path, state = present
            raw_sha256 = state.raw_sha256
            size = state.size
            status = "present"
            restored_from_deletion_generation = intent.restored_from_deletion_generation
        return DocumentControlRecord(
            tenant_id=intent.tenant_id,
            owner_user_id=intent.owner_user_id,
            document_id=intent.document_id,
            relative_path=relative_path,
            raw_sha256=raw_sha256,
            size=size,
            logical_revision=intent.logical_revision,
            projection_generation=intent.projection_generation,
            status=status,
            last_event_id=event.event_id,
            updated_at=event.occurred_at,
            restored_from_deletion_generation=restored_from_deletion_generation,
        )

    @staticmethod
    def _projection_job(intent: DocumentCommitIntent, event: DocumentChangeEvent) -> QueueJob:
        return QueueJob(
            job_id=intent.projection_job_id,
            queue_name="memory_projection",
            action="memory_committed",
            target_uri=MemoryDocumentPathPolicy.document_uri(intent.owner_user_id, intent.document_id),
            payload={
                "schema": "memory_document_projection_v1",
                "tenant_id": intent.tenant_id,
                "owner_user_id": intent.owner_user_id,
                "document_id": intent.document_id,
                "intent_id": intent.intent_id,
                "event_id": event.event_id,
                "edit_kind": event.edit_kind.value,
                "old_relative_path": event.old_relative_path,
                "new_relative_path": event.new_relative_path,
                "before_raw_digest": event.before_raw_digest,
                "after_raw_digest": event.after_raw_digest,
                "logical_revision": event.logical_revision,
                "projection_generation": event.projection_generation,
            },
        )

    def _enqueue_retryable_recovery(
        self,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        intent_id: str,
        exc: BaseException,
    ) -> None:
        """Publish a content-free recovery job only for a durable live intent."""

        if not self._is_retryable_interruption(exc):
            return
        intent = self.control_store.load_intent(tenant_id, owner_user_id, intent_id)
        if intent is None or intent.status in {
            DocumentIntentStatus.COMPLETED,
            DocumentIntentStatus.CONFLICTED,
        }:
            return
        if intent.document_id != document_id:
            raise DocumentControlIntegrityError(
                "recovery job identity differs from its durable document intent"
            )
        self.projection_queue.enqueue(
            QueueJob(
                job_id=f"memory_document_edit_{intent.intent_id}",
                queue_name="memory_document_edit",
                action="recover_document_intent",
                target_uri=MemoryDocumentPathPolicy.document_uri(owner_user_id, document_id),
                payload={
                    "tenant_id": tenant_id,
                    "owner_user_id": owner_user_id,
                    "document_id": document_id,
                    "intent_id": intent.intent_id,
                },
            )
        )

    def _is_retryable_interruption(self, exc: BaseException) -> bool:
        if isinstance(
            exc,
            (
                DocumentCommitConflict,
                DocumentControlIntegrityError,
                DocumentNotFoundError,
                DocumentUnsafeError,
                PermissionError,
                ValueError,
                TypeError,
            ),
        ):
            return False
        if isinstance(exc, (DocumentConflictError, OSError)):
            return True
        explicit = getattr(exc, "retryable", None)
        if isinstance(explicit, bool):
            return explicit
        # Fault hooks model abrupt retryable interruption after a durable
        # stage. Production instances do not install this callback.
        return self.test_hook is not None

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

    def _notify(self, stage: str, intent: DocumentCommitIntent) -> None:
        if self.test_hook is not None:
            try:
                self.test_hook(stage, intent)
            except BaseException as exc:
                self._enqueue_retryable_recovery(
                    intent.tenant_id,
                    intent.owner_user_id,
                    intent.document_id,
                    intent.intent_id,
                    exc,
                )
                raise


def _bounded_text(value: object, label: str, maximum: int) -> str:
    text = str(value or "").strip()
    if not text or len(text) > maximum or any(ord(character) < 32 and character not in "\t" for character in text):
        raise ValueError(f"{label} is empty, too large or contains control characters")
    return text


def _sha256_digest(value: object, label: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


__all__ = [
    "DocumentCommitConflict",
    "DocumentCommitResult",
    "DocumentRecoveryReport",
    "MemoryDocumentCommitter",
]
