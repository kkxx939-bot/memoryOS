"""Phase-aware recovery for ordinary Context operation redo entries."""

from __future__ import annotations

from contextlib import ExitStack

from memoryos.operations.commit.redo_log import RedoIntegrityError
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus


class CommitRecoveryStateMachine:
    @staticmethod
    def resume(
        committer,
        user_id: str,
        operation: ContextOperation,
        phase: str,
        *,
        source_effect: dict | None = None,
        relation_manifest: dict | None = None,
    ) -> bool:
        committer._require_commit_ready(user_id, [operation])
        entry = committer._load_exact_redo_entry(
            user_id,
            operation,
            phase,
            source_effect=source_effect,
            relation_manifest=relation_manifest,
        )
        committer._validate_redo_boundary(
            user_id,
            entry.operation,
            source_effect=entry.source_effect,
            relation_manifest=entry.relation_manifest,
        )
        with ExitStack() as stack:
            guards = [
                stack.enter_context(committer.path_lock.acquire(committer._lock_key(key)))
                for key in committer._regular_lock_keys(entry.operation)
            ]
            with committer.path_lock.fenced(guards):
                return committer._resume_under_guard(
                    user_id,
                    entry.operation,
                    entry.phase,
                    source_effect=entry.source_effect,
                    relation_manifest=entry.relation_manifest,
                    guard=guards[0],
                )

    @staticmethod
    def _resume_unfenced(
        committer,
        user_id: str,
        operation: ContextOperation,
        phase: str,
        *,
        source_effect: dict | None = None,
        relation_manifest: dict | None = None,
    ) -> bool:
        return CommitRecoveryStateMachine.resume(
            committer,
            user_id,
            operation,
            phase,
            source_effect=source_effect,
            relation_manifest=relation_manifest,
        )

    @staticmethod
    def _resume_started_source_effect(
        committer,
        user_id: str,
        operation: ContextOperation,
        *,
        relation_manifest: dict | None,
    ) -> bool:
        manifest = relation_manifest or committer._build_regular_relation_manifest(operation)
        already_applied = False
        try:
            candidate = committer._capture_regular_source_effect(operation, manifest)
            committer._validate_regular_recovery_effect(
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
            committer._prepare_delete_tombstones(operation, trust_durable_binding=True)
        if not already_applied:
            committer._apply_source(operation)
        committer._apply_regular_relation_manifest(operation, manifest)
        effect = committer._capture_regular_source_effect(operation, manifest)
        committer._validate_regular_recovery_effect(
            user_id,
            operation,
            effect,
            relation_manifest=manifest,
        )
        committer.redo.advance(
            operation,
            phase="source_written",
            source_effect=effect,
            relation_manifest=manifest,
        )
        return CommitRecoveryStateMachine._finish_after_source(
            committer,
            user_id,
            operation,
            effect,
            manifest,
        )

    @staticmethod
    def _resume_under_guard(
        committer,
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
            return committer._resume_started_source_effect(
                user_id,
                operation,
                relation_manifest=relation_manifest,
            )
        if phase == "source_written":
            committer._validate_and_restore_regular_recovery_effect(
                user_id,
                operation,
                source_effect,
                relation_manifest,
            )
            assert isinstance(source_effect, dict)
            return CommitRecoveryStateMachine._finish_after_source(
                committer,
                user_id,
                operation,
                source_effect,
                relation_manifest,
            )
        if phase == "index_written":
            committer._validate_and_restore_regular_recovery_effect(
                user_id,
                operation,
                source_effect,
                relation_manifest,
            )
            committer.audit.record(user_id, "context_operation_committed", operation.to_dict())
            committer.redo.advance(operation, phase="audit_written")
            assert isinstance(source_effect, dict)
            return CommitRecoveryStateMachine._finish_after_audit(
                committer, user_id, operation, source_effect, relation_manifest
            )
        if phase == "audit_written":
            committer._validate_and_restore_regular_recovery_effect(
                user_id,
                operation,
                source_effect,
                relation_manifest,
            )
            assert isinstance(source_effect, dict)
            return CommitRecoveryStateMachine._finish_after_audit(
                committer, user_id, operation, source_effect, relation_manifest
            )
        if phase == "diff_written":
            committer._validate_and_restore_regular_recovery_effect(
                user_id,
                operation,
                source_effect,
                relation_manifest,
            )
            diff = committer._ensure_single_operation_diff(user_id, operation)
            committer._write_operation_marker(
                operation,
                source_effect=source_effect,
                relation_manifest=relation_manifest,
                diff=diff,
            )
            committer._refresh_regular_effect_proofs(committer._regular_source_effect_uris(operation))
            committer.redo.commit(operation.operation_id)
            committer._settle_delete_tombstones([operation])
            return True
        if phase == "committed":
            committer.redo.commit(operation.operation_id)
            return True
        raise RedoIntegrityError(f"unsupported ordinary redo phase: {phase}")

    @staticmethod
    def _finish_after_source(
        committer,
        user_id: str,
        operation: ContextOperation,
        source_effect: dict,
        relation_manifest: dict | None,
    ) -> bool:
        committer._apply_index(operation)
        committer.redo.advance(operation, phase="index_written")
        committer.audit.record(user_id, "context_operation_committed", operation.to_dict())
        committer.redo.advance(operation, phase="audit_written")
        return CommitRecoveryStateMachine._finish_after_audit(
            committer, user_id, operation, source_effect, relation_manifest
        )

    @staticmethod
    def _finish_after_audit(
        committer,
        user_id: str,
        operation: ContextOperation,
        source_effect: dict,
        relation_manifest: dict | None,
    ) -> bool:
        operation.status = OperationStatus.COMMITTED
        committer._finalize_single_regular_operation(
            user_id,
            operation,
            source_effect=source_effect,
            relation_manifest=relation_manifest,
        )
        committer._settle_delete_tombstones([operation])
        return True


__all__ = ["CommitRecoveryStateMachine"]
