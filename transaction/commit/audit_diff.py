"""发布稳定且不携带语义正文的操作差异控制记录。"""

from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING

from foundation.clock import utc_now
from foundation.ids import stable_hash
from foundation.integrity import canonical_json
from infrastructure.store.contracts.path_lock import LeaseGuard
from transaction.commit.control import RedoIntegrityError
from transaction.commit.control_record import diff_control_members, diff_control_record
from transaction.model.context_diff import ContextDiff
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_status import OperationStatus

if TYPE_CHECKING:
    from transaction.commit.host import OperationTransactionHost


class CommitAuditDiff:
    def _finalize_regular_diff(
        self: OperationTransactionHost,
        user_id: str,
        committed: list[ContextOperation],
        pending: list[ContextOperation],
        target_rejected: list[ContextOperation],
        conflict_rejected: list[ContextOperation],
        *,
        held_guards: list[LeaseGuard] | None = None,
    ) -> ContextDiff:
        if held_guards is not None:
            with self.path_lock.fenced(held_guards):
                return self._finalize_regular_diff_locked(
                    user_id, committed, pending, target_rejected, conflict_rejected
                )
        with ExitStack() as stack:
            guards = [
                stack.enter_context(self.path_lock.acquire(self._lock_key(key)))
                for key in sorted({key for operation in committed for key in self._regular_lock_keys(operation)})
            ]
            with self.path_lock.fenced(guards):
                return self._finalize_regular_diff_locked(
                    user_id, committed, pending, target_rejected, conflict_rejected
                )

    def _finalize_regular_diff_locked(
        self: OperationTransactionHost,
        user_id: str,
        committed: list[ContextOperation],
        pending: list[ContextOperation],
        target_rejected: list[ContextOperation],
        conflict_rejected: list[ContextOperation],
    ) -> ContextDiff:
        for operation in committed:
            marker = self._operation_marker(operation.operation_id)
            if not marker.exists():
                raise RedoIntegrityError("combined diff contains an unmarked Source effect")
            self._validate_operation_marker(marker, operation)
            self._validate_single_operation_diff(user_id, operation)
        if len(committed) == 1 and not pending and not target_rejected and not conflict_rejected:
            return self._validate_single_operation_diff(user_id, committed[0])
        members = [*committed, *pending, *target_rejected, *conflict_rejected]
        key = stable_hash(
            sorted(
                (
                    operation.operation_id,
                    operation.status.value,
                    self._operation_effect_fingerprint(operation),
                )
                for operation in members
            ),
            length=32,
        )
        diff = ContextDiff(
            user_id=user_id,
            operations=committed,
            pending_operations=pending,
            rejected_operations=[*target_rejected, *conflict_rejected],
            diff_id=f"diff_{key}",
            created_at=min((item.created_at for item in members if item.created_at), default=utc_now()),
        )
        path = self.diff_writer.path(diff.diff_id)
        self._reject_control_symlink(path, "combined diff artifact")
        if not path.exists():
            self.diff_writer.write(
                diff_control_record(
                    diff,
                    tenant_id=self.tenant_id,
                    fingerprint=self._operation_effect_fingerprint,
                )
            )
            return diff
        stored = self.diff_writer.read(diff.diff_id)
        self._validate_diff_control_header(stored, diff)
        if diff_control_members(stored) != self._diff_members(diff):
            raise ValueError("combined diff id conflicts with a different operation set")
        return diff

    def _finalize_single_regular_operation(
        self: OperationTransactionHost,
        user_id: str,
        operation: ContextOperation,
        *,
        source_effect: dict | None,
        relation_manifest: dict | None,
    ) -> ContextDiff:
        diff = self._ensure_single_operation_diff(user_id, operation)
        self.redo.advance(operation, phase="diff_written")
        self._write_operation_marker(
            operation,
            source_effect=source_effect,
            relation_manifest=relation_manifest,
            diff=diff,
        )
        self._refresh_regular_effect_proofs(self._regular_source_effect_uris(operation))
        self.redo.commit(operation.operation_id)
        return diff

    def _ensure_single_operation_diff(
        self: OperationTransactionHost, user_id: str, operation: ContextOperation
    ) -> ContextDiff:
        path = self.diff_writer.path(f"diff_{operation.operation_id}")
        self._reject_control_symlink(path, "single-operation diff artifact")
        if path.exists():
            return self._validate_single_operation_diff(user_id, operation)
        operation.status = OperationStatus.COMMITTED
        diff = ContextDiff(
            user_id=user_id,
            operations=[operation],
            diff_id=f"diff_{operation.operation_id}",
            created_at=operation.created_at,
        )
        self.diff_writer.write(
            diff_control_record(
                diff,
                tenant_id=self.tenant_id,
                fingerprint=self._operation_effect_fingerprint,
            )
        )
        return diff

    def _validate_single_operation_diff(
        self: OperationTransactionHost, user_id: str, operation: ContextOperation
    ) -> ContextDiff:
        path = self.diff_writer.path(f"diff_{operation.operation_id}")
        if not path.exists():
            raise RedoIntegrityError("committed operation has no single-operation diff")
        payload = self.diff_writer.read(f"diff_{operation.operation_id}")
        operation.status = OperationStatus.COMMITTED
        diff = ContextDiff(
            user_id=user_id,
            operations=[operation],
            diff_id=f"diff_{operation.operation_id}",
            created_at=operation.created_at,
        )
        self._validate_diff_control_header(payload, diff)
        if diff_control_members(payload) != self._diff_members(diff):
            raise RedoIntegrityError("single-operation diff conflicts with its committed effect")
        return diff

    def _combine_diffs(self: OperationTransactionHost, user_id: str, diffs: list[ContextDiff]) -> ContextDiff:
        if not diffs:
            return ContextDiff(user_id=user_id, operations=[])
        for diff in diffs:
            if diff.user_id != user_id:
                raise ValueError("committed diff crosses a user boundary")
        if len(diffs) == 1:
            return diffs[0]

        def unique(kind: str) -> list[ContextOperation]:
            values: dict[str, ContextOperation] = {}
            for diff in diffs:
                for operation in getattr(diff, kind):
                    values.setdefault(operation.operation_id, operation)
            return list(values.values())

        members = [
            (kind, item.operation_id, item.status.value, self._operation_effect_fingerprint(item))
            for kind in ("operations", "pending_operations", "rejected_operations")
            for item in unique(kind)
        ]
        combined = ContextDiff(
            user_id=user_id,
            operations=unique("operations"),
            pending_operations=unique("pending_operations"),
            rejected_operations=unique("rejected_operations"),
            diff_id=f"diff_commit_group_{stable_hash([user_id, canonical_json(sorted(members))], length=32)}",
            created_at=min((diff.created_at for diff in diffs if diff.created_at), default=utc_now()),
        )
        path = self.diff_writer.path(combined.diff_id)
        if not path.exists():
            self.diff_writer.write(
                diff_control_record(
                    combined,
                    tenant_id=self.tenant_id,
                    fingerprint=self._operation_effect_fingerprint,
                )
            )
            return combined
        stored = self.diff_writer.read(combined.diff_id)
        self._validate_diff_control_header(stored, combined)
        if diff_control_members(stored) != self._diff_members(combined):
            raise ValueError("combined diff id conflicts with a different operation effect")
        return combined

    def combine_committed_diffs(self: OperationTransactionHost, user_id: str, diffs: list[ContextDiff]) -> ContextDiff:
        return self._combine_diffs(user_id, diffs)

    def _diff_members(self: OperationTransactionHost, diff: ContextDiff) -> tuple[tuple[str, str, str, str], ...]:
        return tuple(
            sorted(
                (
                    kind,
                    operation.operation_id,
                    operation.status.value,
                    self._operation_effect_fingerprint(operation),
                )
                for kind in ("operations", "pending_operations", "rejected_operations")
                for operation in getattr(diff, kind)
            )
        )

    def _validate_diff_control_header(self: OperationTransactionHost, payload: object, diff: ContextDiff) -> None:
        if not isinstance(payload, dict):
            raise ValueError("diff control record must be an object")
        if (
            payload.get("schema_version") != "context_diff_control_v1"
            or payload.get("diff_id") != diff.diff_id
            or payload.get("user_id") != diff.user_id
            or payload.get("tenant_id") != self.tenant_id
            or payload.get("created_at") != diff.created_at
        ):
            raise ValueError("diff control record crosses its operation boundary")

    def _write_recovery_diff(self: OperationTransactionHost, user_id: str, operation: ContextOperation) -> None:
        self._ensure_single_operation_diff(user_id, operation)


__all__ = ["CommitAuditDiff"]
