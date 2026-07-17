"""Implementation component for CommitRecoveryStateMachine.

The public OperationCommitter delegates explicitly to this component so fault
injection hooks remain available on the facade.
"""

from __future__ import annotations

import json
from contextlib import ExitStack

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.transaction.path_lock import LeaseGuard
from memoryos.core.errors import RevisionConflictError
from memoryos.core.integrity import canonical_json
from memoryos.operations.commit.outbox_envelope import (
    OutboxIntegrityError,
    validate_outbox,
)
from memoryos.operations.commit.planning_proof import (
    PlanningProofIntegrityError,
)
from memoryos.operations.commit.receipt import (
    ReceiptIntegrityError,
    load_transaction_receipt,
)
from memoryos.operations.commit.redo_log import RedoIntegrityError
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus


class CommitRecoveryStateMachine:
    """Own the CommitRecoveryStateMachine responsibility of a commit."""

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
        with committer._migration_projection_fence():
            return committer._resume_unfenced(
                user_id,
                operation,
                phase,
                source_effect=source_effect,
                relation_manifest=relation_manifest,
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
        """处理 resume 这一步。"""

        entry = committer._load_exact_redo_entry(
            user_id,
            operation,
            phase,
            source_effect=source_effect,
            relation_manifest=relation_manifest,
        )
        operation = entry.operation
        phase = entry.phase
        source_effect = entry.source_effect
        relation_manifest = entry.relation_manifest
        committer._validate_redo_boundary(
            user_id,
            operation,
            source_effect=source_effect,
            relation_manifest=relation_manifest,
        )
        if operation.payload.get("canonical_memory") is True:
            transaction_id = str(operation.payload.get("transaction_id") or "")
            entries = [
                entry
                for entry in committer.redo.pending_entries()
                if entry.operation.user_id == user_id
                and str(entry.operation.payload.get("transaction_id") or "") == transaction_id
            ]
            if not entries:
                raise RedoIntegrityError("canonical recovery requires the complete durable transaction batch")
            return operation.operation_id in committer.resume_canonical_batch(user_id, entries)
        committer._require_delete_tombstone_capability([operation])
        if operation.action == OperationAction.DELETE:
            already_bound = bool(committer._delete_tombstone_ids(operation))
            tombstone_ids = committer._prepare_delete_tombstones(operation, trust_durable_binding=True)
            if tombstone_ids and (not already_bound or phase in {"started", "begin"}):
                committer.redo.advance(
                    operation,
                    phase="tombstones_enqueued",
                    source_effect=source_effect,
                    relation_manifest=relation_manifest,
                )
                phase = "tombstones_enqueued"
        if phase in {"started", "begin", "tombstones_enqueued"}:
            resumed = committer._resume_started_source_effect(
                user_id,
                operation,
                relation_manifest=relation_manifest,
            )
        else:
            with ExitStack() as locks:
                guards = [
                    locks.enter_context(committer.path_lock.acquire(committer._lock_key(lock_key)))
                    for lock_key in committer._regular_lock_keys(operation)
                ]
                guard = guards[0]
                with committer.path_lock.fenced(guards):
                    resumed = committer._resume_under_guard(
                        user_id,
                        operation,
                        phase,
                        source_effect=source_effect,
                        relation_manifest=relation_manifest,
                        guard=guard,
                    )
        committer._settle_delete_tombstones([operation])
        return resumed

    @staticmethod
    def _resume_started_source_effect(
        committer,
        user_id: str,
        operation: ContextOperation,
        *,
        relation_manifest: dict | None,
    ) -> bool:
        """Adopt a fully matching Source effect from the begin -> phase crash window."""

        with ExitStack() as locks:
            guards = [
                locks.enter_context(committer.path_lock.acquire(committer._lock_key(lock_key)))
                for lock_key in committer._regular_lock_keys(operation)
            ]
            guard = guards[0]
            with committer.path_lock.fenced(guards):
                if relation_manifest is not None:
                    committer._validate_regular_relation_manifest(operation, relation_manifest)
                elif committer.relation_store is not None:
                    raise RedoIntegrityError("regular redo entry is missing its relation manifest")
                replay_source = operation.action == OperationAction.REFRESH_LAYERS
                if not replay_source:
                    effect = committer._capture_regular_source_effect(operation, relation_manifest)
                    try:
                        committer._validate_regular_recovery_effect(
                            user_id,
                            operation,
                            effect,
                            require_relation_presence=False,
                            relation_manifest=relation_manifest,
                        )
                    except RedoIntegrityError:
                        replay_source = True
                if replay_source:
                    committer._apply_source(operation)
                if isinstance(relation_manifest, dict):
                    committer._apply_regular_relation_manifest(operation, relation_manifest)
                effect = committer._capture_regular_source_effect(operation, relation_manifest)
                committer._validate_regular_recovery_effect(
                    user_id,
                    operation,
                    effect,
                    relation_manifest=relation_manifest,
                )
                committer.redo.advance(
                    operation,
                    phase="source_written",
                    source_effect=effect,
                    relation_manifest=relation_manifest,
                )
                return committer._resume_under_guard(
                    user_id,
                    operation,
                    "source_written",
                    source_effect=effect,
                    relation_manifest=relation_manifest,
                    guard=guard,
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
        guard: LeaseGuard,
    ) -> bool:
        del guard
        if phase == "head_published":
            if operation.payload.get("canonical_pending_proposal") is not True:
                raise RedoIntegrityError("regular head-published redo is not a pending lifecycle operation")
            marker = committer._operation_marker(operation.operation_id)
            committer._reject_control_symlink(marker, "pending operation receipt")
            try:
                receipt = load_transaction_receipt(marker)
            except ReceiptIntegrityError as exc:
                raise RedoIntegrityError("head-published pending redo has no valid immutable receipt") from exc
            committer._validate_operation_marker(marker, operation)
            committer._validate_head_published_receipt(marker, receipt)
            committer.redo.commit(operation.operation_id)
            return False
        if phase in {"committed"}:
            if operation.payload.get("canonical_memory") is not True:
                marker = committer._operation_marker(operation.operation_id)
                if not marker.exists():
                    raise RedoIntegrityError("committed redo entry has no operation marker")
                stored = committer._validate_operation_marker(marker, operation)
                if stored.payload.get("canonical_pending_proposal") is True:
                    try:
                        receipt = load_transaction_receipt(marker)
                    except ReceiptIntegrityError as exc:
                        raise RedoIntegrityError("committed pending redo has no valid immutable receipt") from exc
                    # ``committed`` is a legacy post-publication redo phase.
                    # It cannot authorize publication of a missing lifecycle
                    # head: doing so would turn historical receipt replay into
                    # mutable current-state repair.  Only an explicitly
                    # pre-head phase may complete publication.
                    committer._validate_head_published_receipt(marker, receipt)
            committer.redo.commit(operation.operation_id)
            return False
        committer._validate_and_restore_regular_recovery_effect(
            user_id,
            operation,
            source_effect,
            relation_manifest,
        )
        if phase == "source_written":
            committer._apply_index(operation)
            committer.redo.advance(operation, phase="index_written")
            committer.audit.record(user_id, "context_operation_committed", operation.to_dict())
            committer.redo.advance(operation, phase="audit_written")
            committer._finalize_single_regular_operation(
                user_id,
                operation,
                source_effect=source_effect,
                relation_manifest=relation_manifest,
            )
            return True
        if phase == "index_written":
            committer.audit.record(user_id, "context_operation_committed", operation.to_dict())
            committer.redo.advance(operation, phase="audit_written")
            committer._finalize_single_regular_operation(
                user_id,
                operation,
                source_effect=source_effect,
                relation_manifest=relation_manifest,
            )
            return True
        if phase == "audit_written":
            committer._finalize_single_regular_operation(
                user_id,
                operation,
                source_effect=source_effect,
                relation_manifest=relation_manifest,
            )
            return True
        if phase == "diff_written":
            diff = committer._ensure_single_operation_diff(user_id, operation)
            committer._write_operation_marker(
                operation,
                source_effect=source_effect,
                relation_manifest=relation_manifest,
                diff=diff,
            )
            committer.redo.commit(operation.operation_id)
            return True
        return False

    @staticmethod
    def resume_canonical_batch(committer, user_id: str, entries: list) -> list[str]:  # noqa: ANN001
        with committer._migration_projection_fence():
            return committer._resume_canonical_batch_unfenced(user_id, entries)

    @staticmethod
    def _resume_canonical_batch_unfenced(committer, user_id: str, entries: list) -> list[str]:  # noqa: ANN001
        """从事务日志记录的阶段继续完成整批写入。"""

        operations = [entry.operation for entry in entries]
        if not operations:
            return []
        for entry in entries:
            committer._validate_redo_boundary(
                user_id,
                entry.operation,
                source_effect=getattr(entry, "source_effect", None),
                relation_manifest=getattr(entry, "relation_manifest", None),
            )
            committer._validate_canonical_artifact_keys(entry.operation)
        transaction_ids = {str(operation.payload.get("transaction_id", "")) for operation in operations}
        idempotency_keys = {str(operation.payload.get("idempotency_key", "")) for operation in operations}
        if len(transaction_ids) != 1 or "" in transaction_ids or len(idempotency_keys) != 1:
            raise ValueError("canonical recovery requires one complete transaction")
        transaction_id = next(iter(transaction_ids))
        idempotency_key = next(iter(idempotency_keys))
        outbox_path = committer._outbox_path(transaction_id)
        committer._reject_control_symlink(outbox_path, "canonical recovery outbox")
        try:
            prepared = validate_outbox(
                json.loads(outbox_path.read_text(encoding="utf-8")),
                transaction_id=transaction_id,
                idempotency_key=idempotency_key,
                tenant_id=committer.tenant_id,
                user_id=user_id,
                operations=operations,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, OutboxIntegrityError) as exc:
            raise RedoIntegrityError("canonical recovery outbox envelope is invalid") from exc
        try:
            committer.planning_proofs.load_canonical_intent(
                transaction_id,
                operations=operations,
                prepared_intent_digest=str(prepared["prepared_intent_digest"]),
            )
        except PlanningProofIntegrityError as exc:
            raise RedoIntegrityError(
                "canonical recovery outbox is detached from its immutable prepared intent"
            ) from exc
        if prepared["status"] == "aborted":
            for operation in operations:
                committer.redo.commit(operation.operation_id)
            committer.audit.record(
                user_id,
                "canonical_memory_aborted_transaction_recovery_skipped",
                {"transaction_id": transaction_id, "operation_ids": [item.operation_id for item in operations]},
            )
            return []
        expected_operation_ids = [str(item) for item in prepared.get("operation_ids", []) or []]
        by_id = {operation.operation_id: operation for operation in operations}
        prepared_operations = [
            ContextOperation.from_dict(payload)
            for payload in prepared.get("operations", []) or []
            if isinstance(payload, dict)
        ]
        try:
            committer._validate_and_bind_operations(user_id, prepared_operations)
        except ValueError as exc:
            raise RedoIntegrityError("canonical recovery outbox crosses its user or tenant boundary") from exc
        for operation in prepared_operations:
            by_id.setdefault(operation.operation_id, operation)
        if set(expected_operation_ids) != set(by_id):
            raise RuntimeError("canonical recovery outbox is missing transaction operations")
        ordered = [by_id[operation_id] for operation_id in expected_operation_ids]
        head_was_published = any(entry.phase == "head_published" for entry in entries)
        marker = committer._transaction_marker(idempotency_key)
        committer._reject_control_symlink(marker, "canonical transaction receipt")
        if head_was_published:
            if not marker.exists():
                raise RedoIntegrityError(f"head-published redo transaction {transaction_id} has no immutable receipt")
            committer._validate_head_published_receipt(
                marker,
                load_transaction_receipt(marker),
            )
        if prepared["status"] == "committed":
            if not marker.exists():
                raise RedoIntegrityError("committed canonical outbox has no effect marker")
            receipt = load_transaction_receipt(marker)
            # A committed outbox is published strictly after the current
            # head.  It is therefore proof that the transaction already
            # crossed the head boundary, even when a stale/corrupt redo file
            # still claims an earlier phase.  Missing heads at this point are
            # authoritative corruption, never a recoverable pre-head window.
            committer._validate_head_published_receipt(marker, receipt)
            diff = committer._validate_transaction_marker(marker, ordered)
            for operation in ordered:
                committer.redo.commit(operation.operation_id)
            return [operation.operation_id for operation in diff.operations]
        if prepared["status"] not in {"prepared", "source_committed"}:
            raise RedoIntegrityError("canonical recovery outbox is not recoverable")
        try:
            committer._validate_canonical_envelope(user_id, ordered)
        except ValueError as exc:
            raise RedoIntegrityError("canonical recovery operations cross their user or tenant boundary") from exc
        committer._preflight_canonical_revisions(ordered, check_revisions=False)
        committer._validate_authoritative_batch(ordered)
        if not marker.exists():
            committer.final_state_validator.validate(
                ordered,
                tenant_id=committer.tenant_id,
                owner_user_id=user_id,
            )
        relation_manifests = {
            str(effect["operation_id"]): dict(effect.get("relation_manifest", {}) or {})
            for effect in prepared.get("effect_manifests", []) or []
            if isinstance(effect, dict)
        }
        if set(relation_manifests) != set(expected_operation_ids):
            raise RedoIntegrityError("canonical recovery outbox relation manifests are incomplete")
        for operation in ordered:
            committer._validate_canonical_relation_manifest(
                operation,
                relation_manifests[operation.operation_id],
            )
        entries_by_id = {entry.operation.operation_id: entry for entry in entries}
        for operation in ordered:
            entry = entries_by_id[operation.operation_id]
            if canonical_json(getattr(entry, "relation_manifest", None)) != canonical_json(
                relation_manifests[operation.operation_id]
            ):
                raise RedoIntegrityError("canonical redo relation manifest does not match outbox")
            if entry.phase not in {"started", "begin"}:
                committer._validate_canonical_source_effect(
                    operation,
                    getattr(entry, "source_effect", None),
                    relation_manifests[operation.operation_id],
                )
            elif prepared["status"] == "source_committed":
                raise RedoIntegrityError("source_committed outbox has an incomplete redo phase")
        slot_uri = next(
            (
                str(payload.get("uri"))
                for operation in ordered
                if isinstance((payload := operation.payload.get("context_object")), dict)
                and dict(payload.get("metadata", {}) or {}).get("canonical_kind") == "slot"
            ),
            transaction_id,
        )
        lock_keys = {
            f"canonical:{slot_uri}",
            f"canonical-idempotency:{idempotency_key}",
            f"canonical-transaction:{transaction_id}",
            *(
                str(operation.target_uri)
                for operation in ordered
                if committer._canonical_pending_effect(operation) and operation.target_uri
            ),
        }
        with ExitStack() as locks:
            guards: list[LeaseGuard] = []
            for lock_key in sorted(lock_keys):
                guards.append(locks.enter_context(committer.path_lock.acquire(committer._lock_key(lock_key))))
            with committer.path_lock.fenced(guards):
                if marker.exists():
                    receipt = load_transaction_receipt(marker)
                    if head_was_published:
                        committer._validate_head_published_receipt(marker, receipt)
                    else:
                        committer._publish_canonical_current_heads(marker, receipt)
                        committer._mark_current_heads_published(ordered)
                    diff = committer._validate_transaction_marker(marker, ordered)
                    committer._finalize_canonical_outbox(
                        transaction_id,
                        idempotency_key,
                        diff.operations,
                        slot_uri=slot_uri,
                    )
                    for operation in ordered:
                        committer.redo.commit(operation.operation_id)
                    return [operation.operation_id for operation in diff.operations]
            for operation in ordered:
                with committer.path_lock.fenced(guards):
                    payload = operation.payload.get("context_object")
                    if not isinstance(payload, dict):
                        raise ValueError("canonical recovery requires context_object")
                    uri = str(payload["uri"])
                    if committer._canonical_pending_effect(operation):
                        desired_obj = ContextObject.from_dict(payload)
                        try:
                            current = committer.source_store.read_object(uri)
                        except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as exc:
                            raise RevisionConflictError(
                                "canonical recovery cannot find its pending resolution target"
                            ) from exc
                        if canonical_json(current.to_dict()) == canonical_json(desired_obj.to_dict()):
                            committer._validate_existing_canonical_effect(operation)
                        else:
                            committer._validate_pending_lifecycle_cas(operation, validate_resolution_links=False)
                            committer._apply_canonical_source(operation)
                        committer._apply_canonical_relation_manifest(
                            operation,
                            relation_manifests[operation.operation_id],
                        )
                        source_effect = committer._capture_canonical_source_effect(
                            operation,
                            relation_manifests[operation.operation_id],
                        )
                        committer.redo.advance(
                            operation,
                            phase="source_written",
                            source_effect=source_effect,
                            relation_manifest=relation_manifests[operation.operation_id],
                        )
                        committer.audit.record(
                            user_id,
                            "canonical_memory_operation_applied_during_recovery",
                            operation.to_dict(),
                        )
                        committer.redo.advance(operation, phase="audit_written")
                        operation.status = OperationStatus.COMMITTED
                        continue
                    expected = int(operation.payload.get("expected_revision", 0))
                    desired_revision = int(dict(payload.get("metadata", {}) or {}).get("revision", 0))
                    try:
                        actual = int(dict(committer.source_store.read_object(uri).metadata or {}).get("revision", 0))
                    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                        actual = 0
                    if actual == expected:
                        committer._apply_canonical_source(operation)
                    elif actual == desired_revision:
                        committer._validate_existing_canonical_effect(operation)
                    else:
                        raise RevisionConflictError(
                            f"canonical recovery conflict for {uri}: expected {expected} or {desired_revision}, actual {actual}"
                        )
                    committer._apply_canonical_relation_manifest(
                        operation,
                        relation_manifests[operation.operation_id],
                    )
                    source_effect = committer._capture_canonical_source_effect(
                        operation,
                        relation_manifests[operation.operation_id],
                    )
                    committer.redo.advance(
                        operation,
                        phase="source_written",
                        source_effect=source_effect,
                        relation_manifest=relation_manifests[operation.operation_id],
                    )
                    committer.audit.record(
                        user_id, "canonical_memory_operation_applied_during_recovery", operation.to_dict()
                    )
                    committer.redo.advance(operation, phase="audit_written")
                    operation.status = OperationStatus.COMMITTED
            with committer.path_lock.fenced(guards):
                committer._write_outbox_event(
                    transaction_id,
                    idempotency_key,
                    ordered,
                    status="source_committed",
                    relation_manifests=relation_manifests,
                )
                diff = committer._ensure_canonical_transaction_diff(
                    user_id,
                    transaction_id,
                    ordered,
                )
                committer._write_transaction_marker(
                    marker,
                    diff,
                    ordered,
                    relation_manifests=relation_manifests,
                )
                committer._publish_canonical_current_heads(marker, load_transaction_receipt(marker))
                committer._mark_current_heads_published(ordered)
                committer.audit.record(
                    user_id,
                    "canonical_memory_transaction_recovered",
                    {"transaction_id": transaction_id, "operation_ids": [item.operation_id for item in ordered]},
                )
                committer._finalize_canonical_outbox(
                    transaction_id,
                    idempotency_key,
                    ordered,
                    slot_uri=slot_uri,
                )
                for operation in ordered:
                    committer.redo.commit(operation.operation_id)
                return [operation.operation_id for operation in ordered]

    @staticmethod
    def recover_pending_canonical(
        committer,
        user_id: str,
        *,
        commit_group_id: str | None = None,
    ) -> list[str]:
        with committer._migration_projection_fence():
            return committer._recover_pending_canonical_unfenced(
                user_id,
                commit_group_id=commit_group_id,
            )

    @staticmethod
    def _recover_pending_canonical_unfenced(
        committer,
        user_id: str,
        *,
        commit_group_id: str | None = None,
    ) -> list[str]:
        """恢复卡在准备阶段或源数据已写入阶段的记忆事务。"""

        grouped: dict[str, list] = {}
        for entry in committer.redo.pending_entries():
            if (
                entry.operation.user_id != user_id
                or not committer._operation_matches_bound_tenant(entry.operation)
                or entry.operation.payload.get("canonical_memory") is not True
                or (
                    commit_group_id is not None
                    and str(entry.operation.payload.get("commit_group_id") or "") != commit_group_id
                )
            ):
                continue
            transaction_id = str(entry.operation.payload.get("transaction_id", ""))
            grouped.setdefault(transaction_id, []).append(entry)
        recovered = []
        for entries in grouped.values():
            recovered.extend(committer.resume_canonical_batch(user_id, entries))
        return recovered

    @staticmethod
    def recover_pending_regular_memory(
        committer,
        user_id: str,
        *,
        commit_group_id: str,
    ) -> list[str]:
        with committer._migration_projection_fence():
            return committer._recover_pending_regular_memory_unfenced(
                user_id,
                commit_group_id=commit_group_id,
            )

    @staticmethod
    def _recover_pending_regular_memory_unfenced(
        committer,
        user_id: str,
        *,
        commit_group_id: str,
    ) -> list[str]:
        """Finish redo-backed pending-memory effects for one session commit group."""

        recovered: list[str] = []
        for entry in committer.redo.pending_entries():
            operation = entry.operation
            if (
                operation.user_id != user_id
                or not committer._operation_matches_bound_tenant(operation)
                or operation.payload.get("canonical_memory") is True
                or operation.payload.get("canonical_pending_proposal") is not True
                or operation.payload.get("commit_consumer")
                or str(operation.payload.get("commit_group_id") or "") != commit_group_id
            ):
                continue
            if committer.resume(
                user_id,
                operation,
                entry.phase,
                source_effect=entry.source_effect,
                relation_manifest=entry.relation_manifest,
            ):
                recovered.append(operation.operation_id)
        return recovered
