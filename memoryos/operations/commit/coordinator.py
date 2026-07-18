"""Generic commit coordinator for ordinary Context and ActionPolicy effects."""

from __future__ import annotations

from contextlib import ExitStack

from memoryos.operations.commit.redo_log import RedoIntegrityError
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus


class CommitCoordinator:
    @staticmethod
    def commit(committer, user_id: str, operations: list[ContextOperation]) -> ContextDiff:
        return committer._commit_unfenced(user_id, operations)

    @staticmethod
    def _commit_unfenced(committer, user_id: str, operations: list[ContextOperation]) -> ContextDiff:
        committer._require_commit_ready(user_id, operations)
        committer._validate_and_bind_operations(user_id, operations)
        committer._reject_control_symlink(
            committer.artifact_root / "system" / "audit" / f"{user_id}.jsonl",
            "audit control file",
        )
        committer._reject_cross_boundary_redo_collisions(user_id, operations)

        pending_by_id = {entry.operation_id: entry for entry in committer.redo.pending_entries()}
        recovered_diffs: list[ContextDiff] = []
        fresh: list[ContextOperation] = []
        for requested in operations:
            entry = pending_by_id.get(requested.operation_id)
            if entry is None:
                marker = committer._operation_marker(requested.operation_id)
                if marker.exists():
                    stored = committer._validate_operation_marker(marker, requested)
                    recovered_diffs.append(committer._ensure_single_operation_diff(user_id, stored))
                else:
                    fresh.append(requested)
                continue
            if not committer.resume(
                user_id,
                entry.operation,
                entry.phase,
                source_effect=entry.source_effect,
                relation_manifest=entry.relation_manifest,
            ):
                raise RedoIntegrityError("durable ordinary operation did not reach a recoverable phase")
            marker = committer._operation_marker(entry.operation_id)
            stored = committer._validate_operation_marker(marker, entry.operation)
            recovered_diffs.append(committer._ensure_single_operation_diff(user_id, stored))

        resolved: list[ContextOperation] = []
        pending: list[ContextOperation] = []
        target_rejected: list[ContextOperation] = []
        for operation in fresh:
            result = committer.target_resolver.resolve(operation, user_id=user_id)
            if result.resolved:
                resolved.append(result.operation)
            elif result.operation.status == OperationStatus.REJECTED:
                target_rejected.append(result.operation)
            else:
                pending.append(result.operation)
        conflict = committer.conflicts.resolve(committer._coalesce_non_policy_operations(resolved))
        for operation in conflict.rejected:
            operation.status = OperationStatus.REJECTED
        committer._require_delete_tombstone_capability(conflict.accepted)
        committer._preflight_regular_operations(conflict.accepted)

        committed: list[ContextOperation] = []
        lock_keys = sorted(
            {
                key
                for operation in conflict.accepted
                if operation.status != OperationStatus.PENDING
                for key in committer._regular_lock_keys(operation)
            }
        )
        with ExitStack() as stack:
            guards_by_key = {
                key: stack.enter_context(committer.path_lock.acquire(committer._lock_key(key)))
                for key in lock_keys
            }
            guards = list(guards_by_key.values())
            for operation in conflict.accepted:
                if operation.status == OperationStatus.PENDING:
                    pending.append(operation)
                    continue
                operation_guards = [guards_by_key[key] for key in committer._regular_lock_keys(operation)]
                with committer.path_lock.fenced(operation_guards):
                    marker = committer._operation_marker(operation.operation_id)
                    if marker.exists():
                        stored = committer._validate_operation_marker(marker, operation)
                        committer._ensure_single_operation_diff(user_id, stored)
                        committed.append(stored)
                        continue
                    relation_manifest = committer._build_regular_relation_manifest(operation)
                    if operation.action == OperationAction.DELETE:
                        operation.payload.pop("projection_tombstone_ids", None)
                    committer.redo.begin(
                        operation,
                        phase="started",
                        relation_manifest=relation_manifest,
                    )
                    committer._notify("after_redo_begin", operation.operation_id)
                    if operation.action == OperationAction.DELETE:
                        ids = committer._prepare_delete_tombstones(operation)
                        if ids:
                            committer.redo.advance(
                                operation,
                                phase="tombstones_enqueued",
                                relation_manifest=relation_manifest,
                            )
                    committer._apply_source(operation)
                    committer._apply_regular_relation_manifest(operation, relation_manifest)
                    source_effect = committer._capture_regular_source_effect(operation, relation_manifest)
                    committer._validate_regular_recovery_effect(
                        user_id,
                        operation,
                        source_effect,
                        relation_manifest=relation_manifest,
                    )
                    committer.redo.advance(
                        operation,
                        phase="source_written",
                        source_effect=source_effect,
                        relation_manifest=relation_manifest,
                    )
                    committer._notify("after_source_written", operation.operation_id)
                with committer.path_lock.fenced(operation_guards):
                    committer._apply_index(operation)
                    committer.redo.advance(operation, phase="index_written")
                    committer._notify("after_index_written", operation.operation_id)
                with committer.path_lock.fenced(operation_guards):
                    committer.audit.record(user_id, "context_operation_committed", operation.to_dict())
                    committer.redo.advance(operation, phase="audit_written")
                    operation.status = OperationStatus.COMMITTED
                    committer._finalize_single_regular_operation(
                        user_id,
                        operation,
                        source_effect=source_effect,
                        relation_manifest=relation_manifest,
                    )
                    committed.append(operation)
            diff = committer._finalize_regular_diff(
                user_id,
                committed,
                pending,
                target_rejected,
                conflict.rejected,
                held_guards=guards,
            )
            combined = committer._combine_diffs(user_id, [*recovered_diffs, diff])
            committer._settle_delete_tombstones(combined.operations)
        return combined

    @staticmethod
    def _preflight_regular_operations(
        committer,
        operations: list[ContextOperation],
        *,
        validate_resolution_links: bool = True,
        validate_target_state: bool = True,
    ) -> None:
        del validate_resolution_links
        for operation in operations:
            if operation.status == OperationStatus.PENDING:
                continue
            marker = committer._operation_marker(operation.operation_id)
            committer._reject_control_symlink(marker, "operation marker")
            if marker.exists():
                committer._validate_operation_marker(marker, operation)
                continue
            trusted = committer._trusted_inflight_regular_object_effect(operation)
            committer._validate_regular_operation_effect(
                trusted or operation,
                validate_target_state=validate_target_state,
                allow_existing_add=trusted is not None,
            )


__all__ = ["CommitCoordinator"]
