"""按 Redo 阶段恢复普通 Context 操作。"""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING

from transaction.commit.control import RedoIntegrityError
from transaction.commit.control_record import operation_control_record
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_action import OperationAction
from transaction.model.operation_status import OperationStatus

if TYPE_CHECKING:
    from transaction.commit.host import OperationTransactionHost


class CommitRecoveryStateMachine:
    def resume(
        self: OperationTransactionHost,
        user_id: str,
        operation: ContextOperation,
        phase: str,
        *,
        source_effect: dict | None = None,
        relation_manifest: dict | None = None,
    ) -> bool:
        self._require_commit_ready(user_id, [operation])
        entry = self._load_exact_redo_entry(
            user_id,
            operation,
            phase,
            source_effect=source_effect,
            relation_manifest=relation_manifest,
        )
        self._validate_redo_boundary(
            user_id,
            entry.operation,
            source_effect=entry.source_effect,
            relation_manifest=entry.relation_manifest,
        )
        with ExitStack() as stack:
            guards = [
                stack.enter_context(self.path_lock.acquire(self._lock_key(key)))
                for key in self._regular_lock_keys(entry.operation)
            ]
            with self.path_lock.fenced(guards):
                return self._resume_under_guard(
                    user_id,
                    entry.operation,
                    entry.phase,
                    source_effect=entry.source_effect,
                    relation_manifest=entry.relation_manifest,
                    guard=guards[0],
                )

    def _resume_unfenced(
        self: OperationTransactionHost,
        user_id: str,
        operation: ContextOperation,
        phase: str,
        *,
        source_effect: dict | None = None,
        relation_manifest: dict | None = None,
    ) -> bool:
        return self.resume(
            user_id,
            operation,
            phase,
            source_effect=source_effect,
            relation_manifest=relation_manifest,
        )

    def _resume_started_source_effect(
        self: OperationTransactionHost,
        user_id: str,
        operation: ContextOperation,
        *,
        relation_manifest: dict | None,
    ) -> bool:
        manifest = relation_manifest or self._build_regular_relation_manifest(operation)
        already_applied = False
        try:
            candidate = self._capture_regular_source_effect(operation, manifest)
            self._validate_regular_recovery_effect(
                user_id,
                operation,
                candidate,
                require_relation_presence=False,
                relation_manifest=manifest,
            )
            already_applied = True
        except (FileNotFoundError, RedoIntegrityError, ValueError):
            candidate = None
        if operation.action == OperationAction.DELETE:
            self._prepare_delete_tombstones(operation, trust_durable_binding=True)
        if not already_applied:
            self._apply_source(operation)
        self._apply_regular_relation_manifest(operation, manifest)
        effect = self._capture_regular_source_effect(operation, manifest)
        self._validate_regular_recovery_effect(
            user_id,
            operation,
            effect,
            relation_manifest=manifest,
        )
        self.redo.advance(
            operation,
            phase="source_written",
            source_effect=effect,
            relation_manifest=manifest,
        )
        return self._finish_after_source(
            user_id,
            operation,
            effect,
            manifest,
        )

    def _resume_under_guard(
        self: OperationTransactionHost,
        user_id: str,
        operation: ContextOperation,
        phase: str,
        *,
        source_effect: dict | None,
        relation_manifest: dict | None,
        guard,
    ) -> bool:
        del guard
        if phase in {"begin", "started", "tombstones_enqueued"}:
            return self._resume_started_source_effect(
                user_id,
                operation,
                relation_manifest=relation_manifest,
            )
        if phase == "source_written":
            self._validate_and_restore_regular_recovery_effect(
                user_id,
                operation,
                source_effect,
                relation_manifest,
            )
            assert isinstance(source_effect, dict)
            return self._finish_after_source(
                user_id,
                operation,
                source_effect,
                relation_manifest,
            )
        if phase == "index_written":
            self._validate_and_restore_regular_recovery_effect(
                user_id,
                operation,
                source_effect,
                relation_manifest,
            )
            self.audit.record(
                user_id,
                "context_operation_committed",
                operation_control_record(
                    operation,
                    tenant_id=self.tenant_id,
                    fingerprint=self._operation_effect_fingerprint,
                ),
            )
            self.redo.advance(operation, phase="audit_written")
            assert isinstance(source_effect, dict)
            return self._finish_after_audit(user_id, operation, source_effect, relation_manifest)
        if phase == "audit_written":
            self._validate_and_restore_regular_recovery_effect(
                user_id,
                operation,
                source_effect,
                relation_manifest,
            )
            assert isinstance(source_effect, dict)
            return self._finish_after_audit(user_id, operation, source_effect, relation_manifest)
        if phase == "diff_written":
            self._validate_and_restore_regular_recovery_effect(
                user_id,
                operation,
                source_effect,
                relation_manifest,
            )
            diff = self._ensure_single_operation_diff(user_id, operation)
            self._write_operation_marker(
                operation,
                source_effect=source_effect,
                relation_manifest=relation_manifest,
                diff=diff,
            )
            self._refresh_regular_effect_proofs(self._regular_source_effect_uris(operation))
            self.redo.commit(operation.operation_id)
            self._settle_delete_tombstones([operation])
            return True
        raise RedoIntegrityError(f"unsupported ordinary redo phase: {phase}")

    def _finish_after_source(
        self: OperationTransactionHost,
        user_id: str,
        operation: ContextOperation,
        source_effect: dict,
        relation_manifest: dict | None,
    ) -> bool:
        self._apply_index(operation)
        self.redo.advance(operation, phase="index_written")
        self.audit.record(
            user_id,
            "context_operation_committed",
            operation_control_record(
                operation,
                tenant_id=self.tenant_id,
                fingerprint=self._operation_effect_fingerprint,
            ),
        )
        self.redo.advance(operation, phase="audit_written")
        return self._finish_after_audit(user_id, operation, source_effect, relation_manifest)

    def _finish_after_audit(
        self: OperationTransactionHost,
        user_id: str,
        operation: ContextOperation,
        source_effect: dict,
        relation_manifest: dict | None,
    ) -> bool:
        operation.status = OperationStatus.COMMITTED
        self._finalize_single_regular_operation(
            user_id,
            operation,
            source_effect=source_effect,
            relation_manifest=relation_manifest,
        )
        self._settle_delete_tombstones([operation])
        return True


__all__ = ["CommitRecoveryStateMachine"]
