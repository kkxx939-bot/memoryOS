"""普通对象和领域扩展副作用的通用事务协调器。"""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING

from transaction.commit.control import RedoIntegrityError
from transaction.commit.control_record import operation_control_record
from transaction.model.context_diff import ContextDiff
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_action import OperationAction
from transaction.model.operation_status import OperationStatus

if TYPE_CHECKING:
    from transaction.commit.host import OperationTransactionHost


class CommitCoordinator:
    def commit(self: OperationTransactionHost, user_id: str, operations: list[ContextOperation]) -> ContextDiff:
        return self._commit_unfenced(user_id, operations)

    def _commit_unfenced(
        self: OperationTransactionHost, user_id: str, operations: list[ContextOperation]
    ) -> ContextDiff:
        self._require_commit_ready(user_id, operations)
        self._validate_and_bind_operations(user_id, operations)
        self._reject_control_symlink(
            self.artifact_root / "system" / "audit" / f"{user_id}.jsonl",
            "audit control file",
        )
        self._reject_cross_boundary_redo_collisions(user_id, operations)

        pending_by_id = {entry.operation_id: entry for entry in self.redo.pending_entries()}
        recovered_diffs: list[ContextDiff] = []
        fresh: list[ContextOperation] = []
        for requested in operations:
            entry = pending_by_id.get(requested.operation_id)
            if entry is None:
                marker = self._operation_marker(requested.operation_id)
                if marker.exists():
                    stored = self._validate_operation_marker(marker, requested)
                    recovered_diffs.append(self._ensure_single_operation_diff(user_id, stored))
                else:
                    fresh.append(requested)
                continue
            if not self.resume(
                user_id,
                entry.operation,
                entry.phase,
                source_effect=entry.source_effect,
                relation_manifest=entry.relation_manifest,
            ):
                raise RedoIntegrityError("durable ordinary operation did not reach a recoverable phase")
            marker = self._operation_marker(entry.operation_id)
            stored = self._validate_operation_marker(marker, entry.operation)
            recovered_diffs.append(self._ensure_single_operation_diff(user_id, stored))

        resolved: list[ContextOperation] = []
        pending: list[ContextOperation] = []
        target_rejected: list[ContextOperation] = []
        for operation in fresh:
            result = self.target_resolver.resolve(operation, user_id=user_id)
            if result.resolved:
                resolved.append(result.operation)
            elif result.operation.status == OperationStatus.REJECTED:
                target_rejected.append(result.operation)
            else:
                pending.append(result.operation)
        conflict = self.conflicts.resolve(resolved)
        for operation in conflict.rejected:
            if operation.status != OperationStatus.NOOP:
                operation.status = OperationStatus.REJECTED
        self._require_delete_tombstone_capability(conflict.accepted)
        self._preflight_regular_operations(conflict.accepted)

        committed: list[ContextOperation] = []
        lock_keys = sorted(
            {
                key
                for operation in conflict.accepted
                if operation.status != OperationStatus.PENDING
                for key in self._regular_lock_keys(operation)
            }
        )
        with ExitStack() as stack:
            guards_by_key = {key: stack.enter_context(self.path_lock.acquire(self._lock_key(key))) for key in lock_keys}
            guards = list(guards_by_key.values())
            for operation in conflict.accepted:
                if operation.status == OperationStatus.PENDING:
                    pending.append(operation)
                    continue
                operation_guards = [guards_by_key[key] for key in self._regular_lock_keys(operation)]
                with self.path_lock.fenced(operation_guards):
                    marker = self._operation_marker(operation.operation_id)
                    if marker.exists():
                        stored = self._validate_operation_marker(marker, operation)
                        self._ensure_single_operation_diff(user_id, stored)
                        committed.append(stored)
                        continue
                    relation_manifest = self._build_regular_relation_manifest(operation)
                    if operation.action == OperationAction.DELETE:
                        operation.payload.pop("projection_tombstone_ids", None)
                    self.redo.begin(
                        operation,
                        phase="started",
                        relation_manifest=relation_manifest,
                    )
                    self._notify("after_redo_begin", operation.operation_id)
                    if operation.action == OperationAction.DELETE:
                        ids = self._prepare_delete_tombstones(operation)
                        if ids:
                            self.redo.advance(
                                operation,
                                phase="tombstones_enqueued",
                                relation_manifest=relation_manifest,
                            )
                    self._apply_source(operation)
                    self._apply_regular_relation_manifest(operation, relation_manifest)
                    source_effect = self._capture_regular_source_effect(operation, relation_manifest)
                    self._validate_regular_recovery_effect(
                        user_id,
                        operation,
                        source_effect,
                        relation_manifest=relation_manifest,
                    )
                    self.redo.advance(
                        operation,
                        phase="source_written",
                        source_effect=source_effect,
                        relation_manifest=relation_manifest,
                    )
                    self._notify("after_source_written", operation.operation_id)
                with self.path_lock.fenced(operation_guards):
                    self._apply_index(operation)
                    self.redo.advance(operation, phase="index_written")
                    self._notify("after_index_written", operation.operation_id)
                with self.path_lock.fenced(operation_guards):
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
                    operation.status = OperationStatus.COMMITTED
                    self._finalize_single_regular_operation(
                        user_id,
                        operation,
                        source_effect=source_effect,
                        relation_manifest=relation_manifest,
                    )
                    committed.append(operation)
            fresh_result = [*committed, *pending, *target_rejected, *conflict.rejected]
            diffs = list(recovered_diffs)
            if fresh_result:
                diffs.append(
                    self._finalize_regular_diff(
                        user_id,
                        committed,
                        pending,
                        target_rejected,
                        conflict.rejected,
                        held_guards=guards,
                    )
                )
            combined = self._combine_diffs(user_id, diffs)
            self._settle_delete_tombstones(combined.operations)
        return combined

    def _preflight_regular_operations(
        self: OperationTransactionHost,
        operations: list[ContextOperation],
        *,
        validate_resolution_links: bool = True,
        validate_target_state: bool = True,
    ) -> None:
        del validate_resolution_links
        for operation in operations:
            if operation.status == OperationStatus.PENDING:
                continue
            marker = self._operation_marker(operation.operation_id)
            self._reject_control_symlink(marker, "operation marker")
            if marker.exists():
                self._validate_operation_marker(marker, operation)
                continue
            trusted = self._trusted_inflight_regular_object_effect(operation)
            self._validate_regular_operation_effect(
                trusted or operation,
                validate_target_state=validate_target_state,
                allow_existing_add=trusted is not None,
            )


__all__ = ["CommitCoordinator"]
