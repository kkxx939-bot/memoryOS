"""单文档提交的安装、事件、发布队列和尾部完成逻辑。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from infrastructure.store.contracts.queue import QueueJob
from infrastructure.store.memory.control_store import (
    DocumentCommitIntent,
    DocumentControlIntegrityError,
    DocumentControlRecord,
    DocumentIntentStatus,
    DocumentRootIdentityGuard,
)
from memory.core.model import (
    DocumentChangeEvent,
    DocumentEditKind,
    ManagedDocument,
    PresentPath,
    ScanGeneration,
    UnsafePath,
)
from memory.core.structure.path_policy import MemoryDocumentPathPolicy
from memory.ports.document_store import (
    DocumentConflictError,
    DocumentNotFoundError,
    DocumentUnsafeError,
)

if TYPE_CHECKING:
    pass
from memory.commit.document_commit_types import (
    DocumentCommitConflict,
    DocumentCommitResult,
)
from memory.commit.document_recovery import _DocumentCommitRecovery


class _DocumentCommitPublication(_DocumentCommitRecovery):
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
            else:  # pragma: no cover - 枚举取值是封闭集合。
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
        """在修改耐久 Intent、Blob 和正文前绑定新的 CREATE。"""

        if allow_initial_publication and edit_kind is not DocumentEditKind.CREATE:
            raise DocumentControlIntegrityError("only a new CREATE may publish an owner's initial source root identity")
        if edit_kind is DocumentEditKind.CREATE:
            probe = getattr(self.document_store, "probe_write_capabilities", None)
            if not callable(probe):
                raise DocumentControlIntegrityError("document store cannot establish a real owner root before CREATE")
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
            raise DocumentControlIntegrityError("document source root identity changed and requires explicit reset")

    def _verify_existing_root_identity(
        self,
        tenant_id: str,
        owner_user_id: str,
    ) -> None:
        """只校验耐久 Intent 的根身份，不在此处创建授权。"""

        scan = self.document_store.full_scan(tenant_id, owner_user_id)
        self._require_publishable_root_scan(scan)
        durable = self.control_store.load_root_identity(tenant_id, owner_user_id)
        if durable is None:
            raise DocumentControlIntegrityError("durable document intent is missing its source root identity")
        if durable.root_identity != scan.root_identity:
            raise DocumentControlIntegrityError("document source root identity changed and requires explicit reset")

    @staticmethod
    def _require_publishable_root_scan(scan: ScanGeneration) -> None:
        if (
            not scan.complete
            or scan.errors
            or len(scan.root_identity) != 32
            or any(character not in "0123456789abcdef" for character in scan.root_identity)
        ):
            raise DocumentControlIntegrityError("document commit cannot bind an incomplete or unsafe source root scan")

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
        """仅为仍然有效的耐久 Intent 发布不含正文的恢复任务。"""

        if not self._is_retryable_interruption(exc):
            return
        intent = self.control_store.load_intent(tenant_id, owner_user_id, intent_id)
        if intent is None or intent.status in {
            DocumentIntentStatus.COMPLETED,
            DocumentIntentStatus.CONFLICTED,
        }:
            return
        if intent.document_id != document_id:
            raise DocumentControlIntegrityError("recovery job identity differs from its durable document intent")
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
        # 故障钩子模拟耐久阶段后的可重试中断，生产实例不会安装该回调。
        return self.test_hook is not None

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
