"""Implementation component for CommitAuditDiff.

The public OperationCommitter delegates explicitly to this component so fault
injection hooks remain available on the facade.
"""

from __future__ import annotations

import json
from contextlib import ExitStack

from memoryos.contextdb.transaction.path_lock import LeaseGuard
from memoryos.core.clock import utc_now
from memoryos.core.ids import require_safe_path_segment, stable_hash
from memoryos.core.integrity import canonical_json
from memoryos.operations.commit.receipt import (
    TRANSACTION_RECEIPT_SCHEMA_VERSION,
    ReceiptIntegrityError,
    validate_transaction_receipt,
)
from memoryos.operations.commit.redo_log import RedoIntegrityError
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_status import OperationStatus


class CommitAuditDiff:
    """Own the CommitAuditDiff responsibility of a commit."""

    @staticmethod
    def _finalize_regular_diff(
        committer,
        user_id: str,
        committed: list[ContextOperation],
        pending: list[ContextOperation],
        target_rejected: list[ContextOperation],
        conflict_rejected: list[ContextOperation],
        *,
        held_guards: list[LeaseGuard] | None = None,
    ) -> ContextDiff:
        if held_guards is not None:
            with committer.path_lock.fenced(held_guards):
                return committer._finalize_regular_diff_locked(
                    user_id,
                    committed,
                    pending,
                    target_rejected,
                    conflict_rejected,
                )
        guards = []
        with ExitStack() as lock_stack:
            lock_keys = sorted({lock_key for operation in committed for lock_key in committer._regular_lock_keys(operation)})
            for lock_key in lock_keys:
                guards.append(lock_stack.enter_context(committer.path_lock.acquire(committer._lock_key(lock_key))))
            with committer.path_lock.fenced(guards):
                return committer._finalize_regular_diff_locked(
                    user_id,
                    committed,
                    pending,
                    target_rejected,
                    conflict_rejected,
                )

    @staticmethod
    def _finalize_regular_diff_locked(
        committer,
        user_id: str,
        committed: list[ContextOperation],
        pending: list[ContextOperation],
        target_rejected: list[ContextOperation],
        conflict_rejected: list[ContextOperation],
    ) -> ContextDiff:
        for operation in committed:
            marker = committer._operation_marker(operation.operation_id)
            committer._reject_control_symlink(marker, "operation receipt")
            if not marker.exists():
                raise RedoIntegrityError("combined regular diff contains an unmarked Source effect")
            committer._validate_operation_marker(marker, operation)
            committer._validate_single_operation_diff(user_id, operation)
        diff_members = [*committed, *pending, *target_rejected, *conflict_rejected]
        diff_key = stable_hash(
            sorted(
                (
                    operation.operation_id,
                    operation.status.value,
                    committer._operation_effect_fingerprint(operation),
                )
                for operation in diff_members
            ),
            length=32,
        )
        diff = ContextDiff(
            user_id=user_id,
            operations=committed,
            pending_operations=pending,
            rejected_operations=[*target_rejected, *conflict_rejected],
            diff_id=f"diff_{diff_key}",
            created_at=min(
                (operation.created_at for operation in diff_members if operation.created_at), default=utc_now()
            ),
        )
        diff_id = require_safe_path_segment(diff.diff_id, "diff_id")
        diff_path = committer.artifact_root / "system" / "diffs" / f"{diff_id}.json"
        committer._reject_control_symlink(diff_path, "regular diff artifact")
        if diff_path.exists():
            persisted = committer._diff_from_payload(json.loads(diff_path.read_text(encoding="utf-8")))
            requested_ids = {
                "operations": [item.operation_id for item in diff.operations],
                "pending_operations": [item.operation_id for item in diff.pending_operations],
                "rejected_operations": [item.operation_id for item in diff.rejected_operations],
            }
            persisted_ids = {
                "operations": [item.operation_id for item in persisted.operations],
                "pending_operations": [item.operation_id for item in persisted.pending_operations],
                "rejected_operations": [item.operation_id for item in persisted.rejected_operations],
            }
            if requested_ids != persisted_ids:
                raise ValueError("regular diff id conflicts with a different operation set")
            for kind in ("operations", "pending_operations", "rejected_operations"):
                persisted_by_id = {item.operation_id: item for item in getattr(persisted, kind)}
                if any(
                    committer._operation_effect_fingerprint(operation)
                    != committer._operation_effect_fingerprint(persisted_by_id[operation.operation_id])
                    for operation in getattr(diff, kind)
                ):
                    raise ValueError("regular diff conflicts with a different persisted effect")
            diff = persisted
        else:
            committer.diff_writer.write(diff)
        return diff

    @staticmethod
    def _finalize_single_regular_operation(
        committer,
        user_id: str,
        operation: ContextOperation,
        *,
        source_effect: dict | None,
        relation_manifest: dict | None,
    ) -> ContextDiff:
        if operation.payload.get("canonical_pending_proposal") is True:
            committer._bind_pending_receipt_identity(operation)
        diff = committer._ensure_single_operation_diff(user_id, operation)
        committer.redo.advance(operation, phase="diff_written")
        committer._write_operation_marker(
            operation,
            source_effect=source_effect,
            relation_manifest=relation_manifest,
            diff=diff,
        )
        committer._refresh_regular_effect_proofs(committer._regular_source_effect_uris(operation))
        committer.redo.commit(operation.operation_id)
        return diff

    @staticmethod
    def _ensure_single_operation_diff(
        committer,
        user_id: str,
        operation: ContextOperation,
    ) -> ContextDiff:
        operation_id = require_safe_path_segment(operation.operation_id, "operation_id")
        path = committer.artifact_root / "system" / "diffs" / f"diff_{operation_id}.json"
        committer._reject_control_symlink(path, "single-operation diff artifact")
        if path.exists():
            return committer._validate_single_operation_diff(user_id, operation)
        operation.status = OperationStatus.COMMITTED
        diff = ContextDiff(
            user_id=user_id,
            operations=[operation],
            diff_id=f"diff_{operation.operation_id}",
            created_at=operation.created_at,
        )
        committer.diff_writer.write(diff)
        return diff

    @staticmethod
    def _validate_single_operation_diff(
        committer,
        user_id: str,
        operation: ContextOperation,
    ) -> ContextDiff:
        operation_id = require_safe_path_segment(operation.operation_id, "operation_id")
        path = committer.artifact_root / "system" / "diffs" / f"diff_{operation_id}.json"
        committer._reject_control_symlink(path, "single-operation diff artifact")
        if not path.exists():
            raise RedoIntegrityError("committed regular operation has no single-operation diff")
        diff = committer._diff_from_payload(json.loads(path.read_text(encoding="utf-8")))
        if (
            diff.user_id != user_id
            or len(diff.operations) != 1
            or diff.pending_operations
            or diff.rejected_operations
            or diff.operations[0].operation_id != operation.operation_id
            or committer._operation_effect_fingerprint(diff.operations[0]) != committer._operation_effect_fingerprint(operation)
        ):
            raise RedoIntegrityError("single-operation diff conflicts with its committed effect")
        return diff

    @staticmethod
    def _combine_diffs(committer, user_id: str, diffs: list[ContextDiff]) -> ContextDiff:
        for diff in diffs:
            if diff.user_id != user_id:
                raise ValueError("committed diff crosses a user boundary")
            committer._validate_and_bind_operations(
                user_id,
                [*diff.operations, *diff.pending_operations, *diff.rejected_operations],
            )
        if len(diffs) == 1:
            return diffs[0]

        def unique(kind: str) -> list[ContextOperation]:
            by_id: dict[str, ContextOperation] = {}
            for diff in diffs:
                for operation in getattr(diff, kind):
                    by_id.setdefault(operation.operation_id, operation)
            return list(by_id.values())

        effect_members = sorted(
            (
                kind,
                operation.operation_id,
                operation.status.value,
                committer._operation_effect_fingerprint(operation),
            )
            for kind in ("operations", "pending_operations", "rejected_operations")
            for operation in unique(kind)
        )
        combined = ContextDiff(
            user_id=user_id,
            operations=unique("operations"),
            pending_operations=unique("pending_operations"),
            rejected_operations=unique("rejected_operations"),
            diff_id=f"diff_commit_group_{stable_hash([user_id, canonical_json(effect_members)], length=32)}",
            created_at=min((diff.created_at for diff in diffs if diff.created_at), default=utc_now()),
        )
        combined_id = require_safe_path_segment(combined.diff_id, "diff_id")
        path = committer.artifact_root / "system" / "diffs" / f"{combined_id}.json"
        committer._reject_control_symlink(path, "combined diff artifact")
        if not path.exists():
            committer.diff_writer.write(combined)
            return combined
        persisted = committer._diff_from_payload(json.loads(path.read_text(encoding="utf-8")))
        for kind in ("operations", "pending_operations", "rejected_operations"):
            requested = {item.operation_id: item for item in getattr(combined, kind)}
            stored = {item.operation_id: item for item in getattr(persisted, kind)}
            if requested.keys() != stored.keys() or any(
                committer._operation_effect_fingerprint(operation)
                != committer._operation_effect_fingerprint(stored[operation_id])
                for operation_id, operation in requested.items()
            ):
                raise ValueError("combined diff id conflicts with a different operation effect")
        return persisted

    @staticmethod
    def combine_committed_diffs(committer, user_id: str, diffs: list[ContextDiff]) -> ContextDiff:
        """Persist and return one stable diff for already committed effect groups."""

        return committer._combine_diffs(user_id, diffs)

    @staticmethod
    def _ensure_canonical_transaction_diff(
        committer,
        user_id: str,
        transaction_id: str,
        operations: list[ContextOperation],
    ) -> ContextDiff:
        """Create once or validate the immutable diff for one transaction.

        A crash after diff publication but before receipt publication must bind
        the already-published diff.  Reconstructing a new ``ContextDiff`` would
        give it a new timestamp and therefore a different immutable identity.
        """

        transaction_key = require_safe_path_segment(
            transaction_id,
            "canonical transaction_id",
        )
        path = committer.artifact_root / "system" / "diffs" / f"diff_{transaction_key}.json"
        committer._reject_control_symlink(path, "canonical transaction diff")
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("canonical transaction diff must be a JSON object")
                diff = committer._diff_from_payload(payload)
            except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
                raise RedoIntegrityError("canonical transaction diff is unreadable") from exc
            if (
                diff.schema_version != "context_diff_v1"
                or diff.diff_id != f"diff_{transaction_key}"
                or diff.user_id != user_id
                or not diff.created_at
                or diff.pending_operations
                or diff.rejected_operations
                or [item.operation_id for item in diff.operations] != [item.operation_id for item in operations]
                or committer._canonical_transaction_request_fingerprint(diff.operations)
                != committer._canonical_transaction_request_fingerprint(operations)
                or committer._canonical_transaction_effect_fingerprint(diff.operations)
                != committer._canonical_transaction_effect_fingerprint(operations)
            ):
                raise RedoIntegrityError("canonical transaction diff conflicts with its prepared operation set")
            return diff
        diff = ContextDiff(
            user_id=user_id,
            operations=operations,
            diff_id=f"diff_{transaction_key}",
            created_at=min(
                (operation.created_at for operation in operations if operation.created_at),
                default=utc_now(),
            ),
        )
        committer.diff_writer.write(diff)
        return diff

    @staticmethod
    def _diff_from_payload(committer, payload: dict) -> ContextDiff:
        return ContextDiff(
            user_id=str(payload["user_id"]),
            operations=[ContextOperation.from_dict(item) for item in payload.get("operations", [])],
            pending_operations=[ContextOperation.from_dict(item) for item in payload.get("pending_operations", [])],
            rejected_operations=[ContextOperation.from_dict(item) for item in payload.get("rejected_operations", [])],
            diff_id=str(payload.get("diff_id", "")),
            created_at=str(payload.get("created_at", "")),
            schema_version=str(payload.get("schema_version", "context_diff_v1")),
        )

    @staticmethod
    def committed_canonical_diffs(
        committer,
        user_id: str,
        commit_group_id: str,
    ) -> list[ContextDiff]:
        """Load every integrity-checked transaction marker bound to one commit group."""

        root = committer.artifact_root / "system" / "transactions"
        if not root.exists():
            return []
        result: list[ContextDiff] = []
        for path in sorted(root.glob("*.json")):
            if path.is_symlink():
                raise ValueError("canonical transaction receipt cannot be a symbolic link")
            diff = committer._transaction_marker_diff(path)
            if not diff.operations:
                continue
            group_ids = {str(operation.payload.get("commit_group_id") or "") for operation in diff.operations}
            if commit_group_id not in group_ids:
                continue
            if group_ids != {commit_group_id}:
                raise ValueError("canonical transaction marker crosses commit groups")
            if any(
                diff.user_id != user_id
                or operation.user_id != user_id
                or operation.payload.get("canonical_memory") is not True
                for operation in diff.operations
            ):
                raise ValueError("canonical transaction marker crosses a user boundary")
            diff = committer._validate_transaction_marker(path, diff.operations)
            result.append(diff)
        return result

    @staticmethod
    def committed_memory_effect_diffs(
        committer,
        user_id: str,
        commit_group_id: str,
    ) -> list[ContextDiff]:
        """Load marker-backed canonical and pending-memory effects for one group."""

        result = committer.committed_canonical_diffs(user_id, commit_group_id)
        root = committer.artifact_root / "system" / "operations"
        if not root.exists():
            return result
        for path in sorted(root.glob("*.json")):
            if path.is_symlink():
                raise ValueError("pending-memory receipt cannot be a symbolic link")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") == TRANSACTION_RECEIPT_SCHEMA_VERSION:
                try:
                    payload = validate_transaction_receipt(payload)
                except ReceiptIntegrityError as exc:
                    raise ValueError("pending-memory receipt is corrupt") from exc
                receipt_operations = payload.get("operations", [])
                operation_payload = receipt_operations[0] if len(receipt_operations) == 1 else None
            else:
                operation_payload = payload.get("operation")
            if not isinstance(operation_payload, dict):
                continue
            operation = ContextOperation.from_dict(operation_payload)
            if str(operation.payload.get("commit_group_id") or "") != commit_group_id:
                continue
            if operation.payload.get("canonical_pending_proposal") is not True:
                continue
            if operation.payload.get("commit_consumer"):
                continue
            if operation.user_id != user_id:
                raise ValueError("pending-memory operation marker crosses a user boundary")
            committer._validate_and_bind_operations(user_id, [operation])
            stored = committer._validate_operation_marker(path, operation)
            result.append(
                ContextDiff(
                    user_id=user_id,
                    operations=[stored],
                    diff_id=f"diff_{stored.operation_id}",
                    created_at=stored.created_at,
                )
            )
        return result

    @staticmethod
    def _write_recovery_diff(committer, user_id: str, operation: ContextOperation) -> None:
        committer._ensure_single_operation_diff(user_id, operation)
