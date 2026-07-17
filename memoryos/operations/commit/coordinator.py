"""Implementation component for CommitCoordinator.

The public OperationCommitter delegates explicitly to this component so fault
injection hooks remain available on the facade.
"""

from __future__ import annotations

from contextlib import ExitStack

from memoryos.core.errors import RevisionConflictError
from memoryos.operations.commit.planning_proof import (
    PlanningProofIntegrityError,
)
from memoryos.operations.commit.receipt import (
    load_transaction_receipt,
)
from memoryos.operations.commit.redo_log import RedoIntegrityError
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus


class CommitCoordinator:
    """Coordinate generic commit grouping and regular-operation preflight."""

    @staticmethod
    def commit(committer, user_id: str, operations: list[ContextOperation]) -> ContextDiff:
        with committer._migration_projection_fence():
            return committer._commit_unfenced(user_id, operations)

    @staticmethod
    def _commit_unfenced(committer, user_id: str, operations: list[ContextOperation]) -> ContextDiff:
        """执行这一步处理，并保持已有状态约束。"""

        committer._require_commit_ready(user_id, operations)
        committer._validate_and_bind_operations(user_id, operations)
        committer._reject_control_symlink(
            committer.artifact_root / "system" / "audit" / f"{user_id}.jsonl",
            "audit control file",
        )
        committer._reject_cross_boundary_redo_collisions(user_id, operations)
        committer._reject_canonical_regular_bypass(operations)
        canonical = [operation for operation in operations if operation.payload.get("canonical_memory") is True]
        if canonical:
            diffs: list[ContextDiff] = []
            regular = [operation for operation in operations if operation.payload.get("canonical_memory") is not True]
            committer._require_delete_tombstone_capability(regular)
            grouped: dict[str, list[ContextOperation]] = {}
            for operation in canonical:
                transaction_id = str(operation.payload.get("transaction_id", ""))
                grouped.setdefault(transaction_id, []).append(operation)
            # Validate deterministic regular effects and pending lifecycle CAS
            # before the first canonical group can write. Resolution links are
            # intentionally checked only after their canonical Claims commit.
            committer._preflight_regular_operations(
                regular,
                validate_resolution_links=False,
                validate_target_state=False,
            )
            committer._preflight_canonical_groups(user_id, list(grouped.values()))
            try:
                for transaction_operations in grouped.values():
                    diffs.append(committer._commit_canonical_batch(user_id, transaction_operations))
                # Pending/legacy operations are intentionally deferred until
                # every canonical transaction has committed. Keep this inside
                # the same conflict boundary so a regular lifecycle CAS
                # failure still reports the canonical side effects.
                if regular:
                    diffs.append(committer.commit(user_id, regular))
            except RevisionConflictError as exc:
                partials = [*diffs]
                if exc.committed_diff is not None:
                    partials.append(exc.committed_diff)
                partial = committer._combine_diffs(user_id, partials) if partials else None
                raise RevisionConflictError(str(exc), committed_diff=partial) from exc
            return committer._combine_diffs(user_id, diffs)
        pending_regular = {
            entry.operation_id: entry
            for entry in committer.redo.pending_entries()
            if entry.operation.payload.get("canonical_memory") is not True
        }
        recovered_regular_diffs: list[ContextDiff] = []
        unresolved_operations: list[ContextOperation] = []
        for operation in operations:
            entry = pending_regular.get(operation.operation_id)
            if entry is None:
                unresolved_operations.append(operation)
                continue
            committer.resume(
                user_id,
                entry.operation,
                entry.phase,
                source_effect=entry.source_effect,
                relation_manifest=entry.relation_manifest,
            )
            marker = committer._operation_marker(entry.operation_id)
            committer._reject_control_symlink(marker, "operation receipt")
            if not marker.exists():
                raise RedoIntegrityError("regular redo recovery completed without an operation receipt")
            persisted = committer._validate_operation_marker(marker, entry.operation)
            if persisted.payload.get("canonical_pending_proposal") is True:
                committer._validate_head_published_receipt(marker, load_transaction_receipt(marker))
            recovered_regular_diffs.append(committer._ensure_single_operation_diff(user_id, persisted))
        operations = unresolved_operations
        if not operations:
            return committer._combine_diffs(user_id, recovered_regular_diffs)
        resolved_operations: list[ContextOperation] = []
        pending: list[ContextOperation] = []
        target_rejected: list[ContextOperation] = []
        for operation in operations:
            result = committer.target_resolver.resolve(operation, user_id=user_id)
            if result.resolved:
                resolved_operations.append(result.operation)
            elif result.operation.status == OperationStatus.REJECTED:
                target_rejected.append(result.operation)
            else:
                result.operation.status = OperationStatus.PENDING
                pending.append(result.operation)
        conflict_result = committer.conflicts.resolve(committer._coalesce_non_policy_operations(resolved_operations))
        for operation in conflict_result.rejected:
            operation.status = OperationStatus.REJECTED
        committed: list[ContextOperation] = []
        pending_redo = {entry.operation_id: entry for entry in committer.redo.pending_entries()}
        committer._require_delete_tombstone_capability(conflict_result.accepted)
        committer._preflight_regular_operations(conflict_result.accepted)
        with ExitStack() as lock_stack:
            guard_by_key = {
                lock_key: lock_stack.enter_context(committer.path_lock.acquire(committer._lock_key(lock_key)))
                for lock_key in sorted(
                    {
                        lock_key
                        for operation in conflict_result.accepted
                        if operation.status != OperationStatus.PENDING
                        for lock_key in committer._regular_lock_keys(operation)
                    }
                )
            }
            held_guards = list(guard_by_key.values())
            try:
                for operation in conflict_result.accepted:
                    if operation.status == OperationStatus.PENDING:
                        pending.append(operation)
                        continue
                    target_lock_key = operation.target_uri or f"{operation.user_id}:{operation.operation_id}"
                    operation_guards = [guard_by_key[lock_key] for lock_key in committer._regular_lock_keys(operation)]
                    guard = guard_by_key[target_lock_key]
                    with committer.path_lock.fenced(operation_guards):
                        marker = committer._operation_marker(operation.operation_id)
                        committer._reject_control_symlink(marker, "operation receipt")
                        pending_entry = pending_redo.get(operation.operation_id)
                        if pending_entry is not None and pending_entry.phase not in {
                            "started",
                            "begin",
                            "tombstones_enqueued",
                        }:
                            committer._resume_under_guard(
                                user_id,
                                pending_entry.operation,
                                pending_entry.phase,
                                source_effect=pending_entry.source_effect,
                                relation_manifest=pending_entry.relation_manifest,
                                guard=guard,
                            )
                            if marker.exists():
                                persisted = committer._validate_operation_marker(marker, operation)
                                if persisted.payload.get("canonical_pending_proposal") is True:
                                    committer._validate_head_published_receipt(
                                        marker,
                                        load_transaction_receipt(marker),
                                    )
                                committer._ensure_single_operation_diff(user_id, persisted)
                                operation.status = OperationStatus.COMMITTED
                                committed.append(persisted)
                                continue
                        if marker.exists():
                            persisted = committer._validate_operation_marker(marker, operation)
                            if persisted.payload.get("canonical_pending_proposal") is True:
                                committer._validate_head_published_receipt(
                                    marker,
                                    load_transaction_receipt(marker),
                                )
                            committer._ensure_single_operation_diff(user_id, persisted)
                            operation.status = OperationStatus.COMMITTED
                            committed.append(persisted)
                            continue
                        committer._validate_pending_lifecycle_cas(operation)
                        relation_manifest = committer._build_regular_relation_manifest(operation)
                        if operation.payload.get("canonical_pending_proposal") is True:
                            try:
                                committer.planning_proofs.ensure_pending_intent(
                                    operation,
                                    relation_manifest=relation_manifest,
                                )
                            except PlanningProofIntegrityError as exc:
                                raise ValueError("pending lifecycle prepared intent is invalid") from exc
                        if operation.action == OperationAction.DELETE:
                            # This field is an internal outbox binding, never
                            # caller authority.  Persist the semantic intent
                            # first, then replace it only with IDs returned by
                            # the durable Catalog journal.
                            operation.payload.pop("projection_tombstone_ids", None)
                        committer.redo.begin(
                            operation,
                            phase="started",
                            relation_manifest=relation_manifest,
                        )
                        if operation.action == OperationAction.DELETE:
                            tombstone_ids = committer._prepare_delete_tombstones(operation)
                            if tombstone_ids:
                                committer.redo.advance(
                                    operation,
                                    phase="tombstones_enqueued",
                                    relation_manifest=relation_manifest,
                                )
                        committer._apply_source(operation)
                        committer._apply_regular_relation_manifest(operation, relation_manifest)
                        source_effect = committer._capture_regular_source_effect(
                            operation,
                            relation_manifest,
                        )
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
                    with committer.path_lock.fenced(operation_guards):
                        committer._apply_index(operation)
                        committer.redo.advance(operation, phase="index_written")
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
            except RevisionConflictError as exc:
                regular_partials: list[ContextDiff] = list(recovered_regular_diffs)
                if committed or pending or target_rejected or conflict_result.rejected:
                    partial = committer._finalize_regular_diff(
                        user_id,
                        committed,
                        pending,
                        target_rejected,
                        conflict_result.rejected,
                        held_guards=held_guards,
                    )
                    committer._settle_delete_tombstones(partial.operations)
                    regular_partials.append(partial)
                if exc.committed_diff is not None:
                    regular_partials.append(exc.committed_diff)
                committed_diff = committer._combine_diffs(user_id, regular_partials) if regular_partials else None
                raise RevisionConflictError(str(exc), committed_diff=committed_diff) from exc
            diff = committer._finalize_regular_diff(
                user_id,
                committed,
                pending,
                target_rejected,
                conflict_result.rejected,
                held_guards=held_guards,
            )
            committer._settle_delete_tombstones(diff.operations)
            return committer._combine_diffs(user_id, [*recovered_regular_diffs, diff])

    @staticmethod
    def _preflight_regular_operations(
        committer,
        operations: list[ContextOperation],
        *,
        validate_resolution_links: bool = True,
        validate_target_state: bool = True,
    ) -> None:
        """Parse every deterministic effect before any regular write occurs."""

        for operation in operations:
            if operation.payload.get("canonical_memory") is True or operation.status == OperationStatus.PENDING:
                continue
            if operation.payload.get("canonical_pending_proposal") is True:
                committer._bind_pending_receipt_identity(operation)
            committer._reject_control_symlink(
                committer.artifact_root / "system" / "diffs" / f"diff_{operation.operation_id}.json",
                "single-operation diff artifact",
            )
            marker = committer._operation_marker(operation.operation_id)
            committer._reject_control_symlink(marker, "operation receipt")
            if marker.exists():
                committer._validate_operation_marker(marker, operation)
                continue
            trusted_inflight = committer._trusted_inflight_regular_object_effect(operation)
            committer._validate_regular_operation_effect(
                trusted_inflight or operation,
                validate_target_state=validate_target_state,
                allow_existing_add=trusted_inflight is not None,
            )
            if trusted_inflight is not None:
                continue
            committer._validate_pending_lifecycle_cas(
                operation,
                validate_resolution_links=validate_resolution_links,
            )
            if operation.payload.get("canonical_pending_proposal") is True:
                committer._ensure_pending_planning_digest(operation)
