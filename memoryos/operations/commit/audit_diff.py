"""Stable diff publication for ordinary operation commits."""

from __future__ import annotations

import json
from contextlib import ExitStack

from memoryos.contextdb.transaction.path_lock import LeaseGuard
from memoryos.core.clock import utc_now
from memoryos.core.ids import stable_hash
from memoryos.core.integrity import canonical_json
from memoryos.operations.commit.redo_log import RedoIntegrityError
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_status import OperationStatus


class CommitAuditDiff:
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
                    user_id, committed, pending, target_rejected, conflict_rejected
                )
        with ExitStack() as stack:
            guards = [
                stack.enter_context(committer.path_lock.acquire(committer._lock_key(key)))
                for key in sorted(
                    {key for operation in committed for key in committer._regular_lock_keys(operation)}
                )
            ]
            with committer.path_lock.fenced(guards):
                return committer._finalize_regular_diff_locked(
                    user_id, committed, pending, target_rejected, conflict_rejected
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
            if not marker.exists():
                raise RedoIntegrityError("combined diff contains an unmarked Source effect")
            committer._validate_operation_marker(marker, operation)
            committer._validate_single_operation_diff(user_id, operation)
        members = [*committed, *pending, *target_rejected, *conflict_rejected]
        key = stable_hash(
            sorted(
                (
                    operation.operation_id,
                    operation.status.value,
                    committer._operation_effect_fingerprint(operation),
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
        path = committer.artifact_root / "system" / "diffs" / f"{diff.diff_id}.json"
        committer._reject_control_symlink(path, "combined diff artifact")
        if not path.exists():
            committer.diff_writer.write(diff)
            return diff
        stored = committer._diff_from_payload(json.loads(path.read_text(encoding="utf-8")))
        if CommitAuditDiff._diff_members(committer, stored) != CommitAuditDiff._diff_members(committer, diff):
            raise ValueError("combined diff id conflicts with a different operation set")
        return stored

    @staticmethod
    def _finalize_single_regular_operation(
        committer,
        user_id: str,
        operation: ContextOperation,
        *,
        source_effect: dict | None,
        relation_manifest: dict | None,
    ) -> ContextDiff:
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
    def _ensure_single_operation_diff(committer, user_id: str, operation: ContextOperation) -> ContextDiff:
        path = committer.artifact_root / "system" / "diffs" / f"diff_{operation.operation_id}.json"
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
    def _validate_single_operation_diff(committer, user_id: str, operation: ContextOperation) -> ContextDiff:
        path = committer.artifact_root / "system" / "diffs" / f"diff_{operation.operation_id}.json"
        if not path.exists():
            raise RedoIntegrityError("committed operation has no single-operation diff")
        diff = committer._diff_from_payload(json.loads(path.read_text(encoding="utf-8")))
        if (
            diff.user_id != user_id
            or len(diff.operations) != 1
            or diff.pending_operations
            or diff.rejected_operations
            or diff.operations[0].operation_id != operation.operation_id
            or committer._operation_effect_fingerprint(diff.operations[0])
            != committer._operation_effect_fingerprint(operation)
        ):
            raise RedoIntegrityError("single-operation diff conflicts with its committed effect")
        return diff

    @staticmethod
    def _combine_diffs(committer, user_id: str, diffs: list[ContextDiff]) -> ContextDiff:
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
            (kind, item.operation_id, item.status.value, committer._operation_effect_fingerprint(item))
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
        path = committer.artifact_root / "system" / "diffs" / f"{combined.diff_id}.json"
        if not path.exists():
            committer.diff_writer.write(combined)
            return combined
        stored = committer._diff_from_payload(json.loads(path.read_text(encoding="utf-8")))
        if CommitAuditDiff._diff_members(committer, stored) != CommitAuditDiff._diff_members(committer, combined):
            raise ValueError("combined diff id conflicts with a different operation effect")
        return stored

    @staticmethod
    def combine_committed_diffs(committer, user_id: str, diffs: list[ContextDiff]) -> ContextDiff:
        return committer._combine_diffs(user_id, diffs)

    @staticmethod
    def _diff_members(committer, diff: ContextDiff) -> tuple:
        return tuple(
            sorted(
                (
                    kind,
                    operation.operation_id,
                    committer._operation_effect_fingerprint(operation),
                )
                for kind in ("operations", "pending_operations", "rejected_operations")
                for operation in getattr(diff, kind)
            )
        )

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
    def _write_recovery_diff(committer, user_id: str, operation: ContextOperation) -> None:
        committer._ensure_single_operation_diff(user_id, operation)


__all__ = ["CommitAuditDiff"]
