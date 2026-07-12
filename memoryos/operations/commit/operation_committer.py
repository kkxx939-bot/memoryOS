"""操作提交里的操作提交。"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import ExitStack
from pathlib import Path

from memoryos.action_policy.model.action_policy import ActionPolicy
from memoryos.action_policy.model.reward_signal import PenaltySignal, RewardSignal
from memoryos.action_policy.update.action_policy_updater import ActionPolicyUpdater
from memoryos.contextdb.layers.layer_refresher import LayerRefresher
from memoryos.contextdb.model.context_layer import ContextLayers
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.store.local_stores import InMemoryLockStore
from memoryos.contextdb.store.source_store import (
    IndexStore,
    LockStore,
    QueueJob,
    QueueStore,
    RelationStore,
    SourceStore,
)
from memoryos.contextdb.transaction.path_lock import PathLock
from memoryos.core.ids import stable_hash
from memoryos.core.time import utc_now
from memoryos.memory.canonical.event import canonical_json, resolve_content_path
from memoryos.memory.canonical.evidence import evidence_hash
from memoryos.memory.canonical.identity import IDENTITY_ALGORITHM_V2, canonical_text
from memoryos.memory.canonical.proposal import PENDING_PROPOSAL_TRANSITIONS, PendingMemoryProposal
from memoryos.memory.canonical.scope import ScopeRef
from memoryos.memory.canonical.transaction import RevisionConflictError
from memoryos.operations.commit.audit_writer import AuditWriter
from memoryos.operations.commit.diff_writer import DiffWriter
from memoryos.operations.commit.operation_coalescer import OperationCoalescer
from memoryos.operations.commit.redo_log import RedoLog
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus
from memoryos.operations.resolver.conflict_resolver import ConflictResolver
from memoryos.operations.resolver.target_resolver import TargetResolver


class OperationCommitter:
    """负责加锁、版本校验、批量提交、故障恢复和 Outbox 落盘。"""

    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        root: str,
        lock_store: LockStore | None = None,
        relation_store: RelationStore | None = None,
        queue_store: QueueStore | None = None,
        target_resolver: TargetResolver | None = None,
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.relation_store = relation_store
        self.queue_store = queue_store
        self.root = Path(root)
        self.coalescer = OperationCoalescer()
        self.conflicts = ConflictResolver()
        self.target_resolver = target_resolver or TargetResolver(index_store, source_store=source_store)
        self.redo = RedoLog(root)
        self.diff_writer = DiffWriter(root)
        self.audit = AuditWriter(root)
        self.path_lock = PathLock(lock_store or InMemoryLockStore())
        self.action_policy_updater = ActionPolicyUpdater()

    def commit(self, user_id: str, operations: list[ContextOperation]) -> ContextDiff:
        """执行这一步处理，并保持已有状态约束。"""

        canonical = [operation for operation in operations if operation.payload.get("canonical_memory") is True]
        if canonical:
            diffs: list[ContextDiff] = []
            regular = [operation for operation in operations if operation.payload.get("canonical_memory") is not True]
            grouped: dict[str, list[ContextOperation]] = {}
            for operation in canonical:
                transaction_id = str(operation.payload.get("transaction_id", ""))
                grouped.setdefault(transaction_id, []).append(operation)
            # Validate deterministic regular effects and pending lifecycle CAS
            # before the first canonical group can write. Resolution links are
            # intentionally checked only after their canonical Claims commit.
            self._preflight_regular_operations(
                regular,
                validate_resolution_links=False,
                validate_target_state=False,
            )
            self._preflight_canonical_groups(user_id, list(grouped.values()))
            try:
                for transaction_operations in grouped.values():
                    diffs.append(self._commit_canonical_batch(user_id, transaction_operations))
                # Pending/legacy operations are intentionally deferred until
                # every canonical transaction has committed. Keep this inside
                # the same conflict boundary so a regular lifecycle CAS
                # failure still reports the canonical side effects.
                if regular:
                    diffs.append(self.commit(user_id, regular))
            except RevisionConflictError as exc:
                partials = [*diffs]
                if exc.committed_diff is not None:
                    partials.append(exc.committed_diff)
                partial = self._combine_diffs(user_id, partials) if partials else None
                raise RevisionConflictError(str(exc), committed_diff=partial) from exc
            return self._combine_diffs(user_id, diffs)
        resolved_operations: list[ContextOperation] = []
        pending: list[ContextOperation] = []
        target_rejected: list[ContextOperation] = []
        for operation in operations:
            result = self.target_resolver.resolve(operation, user_id=user_id)
            if result.resolved:
                resolved_operations.append(result.operation)
            elif result.operation.status == OperationStatus.REJECTED:
                target_rejected.append(result.operation)
            else:
                result.operation.status = OperationStatus.PENDING
                pending.append(result.operation)
        conflict_result = self.conflicts.resolve(self._coalesce_non_policy_operations(resolved_operations))
        for operation in conflict_result.rejected:
            operation.status = OperationStatus.REJECTED
        committed: list[ContextOperation] = []
        pending_redo = {entry.operation_id: entry for entry in self.redo.pending_entries()}
        self._preflight_regular_operations(conflict_result.accepted)
        try:
            for operation in conflict_result.accepted:
                if operation.status == OperationStatus.PENDING:
                    pending.append(operation)
                    continue
                lock_key = operation.target_uri or f"{operation.user_id}:{operation.operation_id}"
                with self.path_lock.acquire(lock_key):
                    marker = self._operation_marker(operation.operation_id)
                    if marker.exists():
                        persisted = self._validate_operation_marker(marker, operation)
                        operation.status = OperationStatus.COMMITTED
                        committed.append(persisted)
                        continue
                    pending_entry = pending_redo.get(operation.operation_id)
                    if pending_entry is not None and pending_entry.phase not in {"started", "begin"}:
                        self.resume(user_id, pending_entry.operation, pending_entry.phase)
                        if marker.exists():
                            persisted = self._validate_operation_marker(marker, operation)
                            operation.status = OperationStatus.COMMITTED
                            committed.append(persisted)
                            continue
                    self._validate_pending_lifecycle_cas(operation)
                    self.redo.begin(operation, phase="started")
                    self._apply_source(operation)
                    self.redo.advance(operation, phase="source_written")
                    self._apply_index(operation)
                    self.redo.advance(operation, phase="index_written")
                    self.audit.record(user_id, "context_operation_committed", operation.to_dict())
                    self.redo.advance(operation, phase="audit_written")
                    operation.status = OperationStatus.COMMITTED
                committed.append(operation)
        except RevisionConflictError as exc:
            regular_partials: list[ContextDiff] = []
            if committed or pending or target_rejected or conflict_result.rejected:
                regular_partials.append(
                    self._finalize_regular_diff(
                        user_id,
                        committed,
                        pending,
                        target_rejected,
                        conflict_result.rejected,
                    )
                )
            if exc.committed_diff is not None:
                regular_partials.append(exc.committed_diff)
            committed_diff = self._combine_diffs(user_id, regular_partials) if regular_partials else None
            raise RevisionConflictError(str(exc), committed_diff=committed_diff) from exc
        return self._finalize_regular_diff(
            user_id,
            committed,
            pending,
            target_rejected,
            conflict_result.rejected,
        )

    def _finalize_regular_diff(
        self,
        user_id: str,
        committed: list[ContextOperation],
        pending: list[ContextOperation],
        target_rejected: list[ContextOperation],
        conflict_rejected: list[ContextOperation],
    ) -> ContextDiff:
        diff_members = [*committed, *pending, *target_rejected, *conflict_rejected]
        diff_key = stable_hash(
            sorted((operation.operation_id, operation.status.value) for operation in diff_members),
            length=32,
        )
        diff = ContextDiff(
            user_id=user_id,
            operations=committed,
            pending_operations=pending,
            rejected_operations=[*target_rejected, *conflict_rejected],
            diff_id=f"diff_{diff_key}",
            created_at=min((operation.created_at for operation in diff_members if operation.created_at), default=utc_now()),
        )
        diff_path = self.root / "system" / "diffs" / f"{diff.diff_id}.json"
        if diff_path.exists():
            persisted = self._diff_from_payload(json.loads(diff_path.read_text(encoding="utf-8")))
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
            persisted_by_id = {item.operation_id: item for item in persisted.operations}
            if any(
                self._operation_effect_fingerprint(operation)
                != self._operation_effect_fingerprint(persisted_by_id[operation.operation_id])
                for operation in diff.operations
            ):
                raise ValueError("regular diff conflicts with a different persisted effect")
            diff = persisted
        else:
            self.diff_writer.write(diff)
        for operation in committed:
            self._write_operation_marker(operation)
            self.redo.advance(operation, phase="diff_written")
            self.redo.commit(operation.operation_id)
        return diff

    def _combine_diffs(self, user_id: str, diffs: list[ContextDiff]) -> ContextDiff:
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
                self._operation_effect_fingerprint(operation),
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
        path = self.root / "system" / "diffs" / f"{combined.diff_id}.json"
        if not path.exists():
            self.diff_writer.write(combined)
            return combined
        persisted = self._diff_from_payload(json.loads(path.read_text(encoding="utf-8")))
        for kind in ("operations", "pending_operations", "rejected_operations"):
            requested = {item.operation_id: item for item in getattr(combined, kind)}
            stored = {item.operation_id: item for item in getattr(persisted, kind)}
            if requested.keys() != stored.keys() or any(
                self._operation_effect_fingerprint(operation)
                != self._operation_effect_fingerprint(stored[operation_id])
                for operation_id, operation in requested.items()
            ):
                raise ValueError("combined diff id conflicts with a different operation effect")
        return persisted

    def combine_committed_diffs(self, user_id: str, diffs: list[ContextDiff]) -> ContextDiff:
        """Persist and return one stable diff for already committed effect groups."""

        return self._combine_diffs(user_id, diffs)

    def _commit_canonical_batch(self, user_id: str, operations: list[ContextOperation]) -> ContextDiff:
        if not operations:
            return ContextDiff(user_id=user_id)
        self._validate_canonical_envelope(user_id, operations)
        transaction_ids = {str(operation.payload.get("transaction_id", "")) for operation in operations}
        idempotency_keys = {str(operation.payload.get("idempotency_key", "")) for operation in operations}
        if len(transaction_ids) != 1 or "" in transaction_ids or len(idempotency_keys) != 1 or "" in idempotency_keys:
            raise ValueError("canonical batch requires one transaction_id and idempotency_key")
        transaction_id = next(iter(transaction_ids))
        idempotency_key = next(iter(idempotency_keys))
        completed = self._transaction_marker(idempotency_key)
        if completed.exists():
            diff = self._validate_transaction_marker(completed, operations)
            self._finalize_canonical_outbox(transaction_id, idempotency_key, diff.operations)
            return diff

        slot_uri = next(
            (
                str(payload.get("uri"))
                for operation in operations
                if isinstance((payload := operation.payload.get("context_object")), dict)
                and dict(payload.get("metadata", {}) or {}).get("canonical_kind") == "slot"
            ),
            transaction_id,
        )
        lock_keys = {
            f"canonical:{slot_uri}",
            *(
                str(operation.target_uri)
                for operation in operations
                if operation.payload.get("canonical_pending_resolution") is True and operation.target_uri
            ),
        }
        with ExitStack() as locks:
            for lock_key in sorted(lock_keys):
                locks.enter_context(self.path_lock.acquire(lock_key))
            if completed.exists():
                diff = self._validate_transaction_marker(completed, operations)
                self._finalize_canonical_outbox(transaction_id, idempotency_key, diff.operations)
                return diff
            self._preflight_canonical_revisions(operations)
            self._validate_authoritative_batch(operations)
            backups = self._capture_canonical_state(operations)
            committed: list[ContextOperation] = []
            self._write_outbox_event(
                transaction_id,
                idempotency_key,
                operations,
                status="prepared",
                before_images=backups,
            )
            for operation in operations:
                self.redo.begin(operation, phase="started")
            try:
                for operation in operations:
                    self._apply_canonical_source(operation)
                    self.redo.advance(operation, phase="source_written")
                    self.audit.record(user_id, "canonical_memory_operation_applied", operation.to_dict())
                    self.redo.advance(operation, phase="audit_written")
                    operation.status = OperationStatus.COMMITTED
                    committed.append(operation)
                self._write_outbox_event(
                    transaction_id,
                    idempotency_key,
                    committed,
                    status="source_committed",
                    before_images=backups,
                )
            except Exception:
                self._restore_canonical_state(backups)
                self._write_outbox_event(
                    transaction_id,
                    idempotency_key,
                    operations,
                    status="aborted",
                )
                for operation in operations:
                    self.redo.commit(operation.operation_id)
                self.audit.record(
                    user_id,
                    "canonical_memory_transaction_rolled_back",
                    {"transaction_id": transaction_id, "operation_ids": [item.operation_id for item in operations]},
                )
                raise
            diff = ContextDiff(
                user_id=user_id,
                operations=committed,
                diff_id=f"diff_{transaction_id}",
            )
            self.diff_writer.write(diff)
            self._write_transaction_marker(completed, diff, committed)
            self.audit.record(
                user_id,
                "canonical_memory_transaction_committed",
                {"transaction_id": transaction_id, "operation_ids": [item.operation_id for item in committed]},
            )
            self._finalize_canonical_outbox(transaction_id, idempotency_key, committed, slot_uri=slot_uri)
            for operation in committed:
                self.redo.commit(operation.operation_id)
            return diff

    def _preflight_canonical_groups(
        self,
        user_id: str,
        groups: list[list[ContextOperation]],
    ) -> None:
        """Validate every group before the first group can create a side effect."""

        virtual_revisions: dict[str, int] = {}
        idempotency_transactions: dict[str, str] = {}
        for operations in groups:
            if not operations:
                continue
            self._validate_canonical_envelope(user_id, operations)
            transaction_ids = {str(operation.payload.get("transaction_id", "")) for operation in operations}
            idempotency_keys = {str(operation.payload.get("idempotency_key", "")) for operation in operations}
            if (
                len(transaction_ids) != 1
                or "" in transaction_ids
                or len(idempotency_keys) != 1
                or "" in idempotency_keys
            ):
                raise ValueError("canonical batch requires one transaction_id and idempotency_key")
            transaction_id = next(iter(transaction_ids))
            idempotency_key = next(iter(idempotency_keys))
            existing_transaction = idempotency_transactions.setdefault(idempotency_key, transaction_id)
            if existing_transaction != transaction_id:
                raise ValueError("canonical idempotency key cannot identify multiple transactions")
            self._canonical_transaction_request_fingerprint(operations)
            self._canonical_transaction_effect_fingerprint(operations)
            marker = self._transaction_marker(idempotency_key)
            if marker.exists():
                self._validate_transaction_marker(marker, operations)
                continue
            for operation in operations:
                if operation.payload.get("canonical_pending_resolution") is True:
                    self._validate_pending_lifecycle_cas(operation, validate_resolution_links=False)
            self._validate_pending_resolution_batch(operations)
            self._preflight_canonical_revisions(operations, check_revisions=False)
            self._validate_authoritative_batch(operations)
            for operation in operations:
                if operation.payload.get("canonical_pending_resolution") is True:
                    continue
                object_payload = operation.payload.get("context_object")
                assert isinstance(object_payload, dict)
                uri = str(object_payload["uri"])
                if uri not in virtual_revisions:
                    try:
                        current = self.source_store.read_object(uri)
                        virtual_revisions[uri] = int(dict(current.metadata or {}).get("revision", 0))
                    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                        virtual_revisions[uri] = 0
                expected = int(operation.payload.get("expected_revision", 0))
                if virtual_revisions[uri] != expected:
                    raise RevisionConflictError(
                        f"revision conflict for {uri}: expected {expected}, actual {virtual_revisions[uri]}"
                    )
                virtual_revisions[uri] = int(dict(object_payload.get("metadata", {}) or {}).get("revision", 0))

    def _validate_canonical_envelope(self, user_id: str, operations: list[ContextOperation]) -> None:
        """Validate immutable ownership boundaries before any marker fast path."""

        if not user_id:
            raise ValueError("canonical commit requires a user_id")
        for operation in operations:
            if operation.user_id != user_id:
                raise ValueError("canonical operation user does not match commit user")
            object_payload = operation.payload.get("context_object")
            if not isinstance(object_payload, dict):
                raise ValueError("canonical operation requires context_object")
            obj = ContextObject.from_dict(object_payload)
            target_uri = str(operation.target_uri or "")
            if not target_uri or target_uri != obj.uri:
                raise ValueError("canonical target_uri does not match context_object URI")
            parsed = ContextURI.parse(obj.uri)
            if obj.owner_user_id != user_id:
                raise ValueError("canonical context object tenant or owner does not match commit user")
            if operation.context_type != ContextType.MEMORY or obj.context_type != operation.context_type:
                raise ValueError("canonical context type must be memory and match its operation")
            operation_tenant = str(operation.payload.get("tenant_id") or "default")
            object_tenant = str(obj.tenant_id or "default")
            if object_tenant != operation_tenant:
                raise ValueError("canonical context object tenant does not match operation tenant")
            metadata = dict(obj.metadata or {})
            if operation.payload.get("canonical_pending_resolution") is True:
                if (
                    operation.action != OperationAction.UPDATE
                    or operation.payload.get("pending_lifecycle_resolution") is not True
                    or operation.payload.get("pending_lifecycle_transition") is not True
                    or metadata.get("canonical_kind") != "pending_proposal"
                    or obj.schema_version != PendingMemoryProposal.SCHEMA_VERSION
                ):
                    raise ValueError("canonical pending resolution envelope is invalid")
                continue
            scope = dict(metadata.get("scope", {}) or {})
            subject_payload = scope.get("canonical_subject")
            if not isinstance(subject_payload, dict):
                raise ValueError("canonical target URI requires a canonical subject")
            subject = ScopeRef.from_dict(subject_payload)
            expected_storage_owner = (
                user_id
                if subject.kind == "principal" and canonical_text(subject.id) == canonical_text(user_id)
                else f"subject_{stable_hash([operation_tenant, subject.key], length=20)}"
            )
            if parsed.authority != "user" or parsed.user_id != expected_storage_owner:
                raise ValueError("canonical target URI owner does not match canonical subject boundary")
            visibility = dict(scope.get("visibility", {}) or {})
            if str(visibility.get("tenant_id") or "default") != operation_tenant:
                raise ValueError("canonical visibility scope crosses the operation tenant")
            principals = {
                str(item) for item in dict(scope.get("authority", {}) or {}).get("principal_ids", []) or []
            }
            if principals and user_id not in principals:
                raise ValueError("canonical assertion scope does not authorize the commit user")
            if int(operation.payload.get("expected_revision", 0) or 0) > 0:
                self._validate_existing_canonical_boundary(obj)

    def _validate_existing_canonical_boundary(self, desired: ContextObject) -> None:
        try:
            current = self.source_store.read_object(desired.uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return
        desired_metadata = dict(desired.metadata or {})
        current_metadata = dict(current.metadata or {})
        if (
            current.owner_user_id != desired.owner_user_id
            or str(current.tenant_id or "default") != str(desired.tenant_id or "default")
            or current.context_type != desired.context_type
        ):
            raise ValueError("canonical UPDATE cannot change owner, tenant, or context type")
        for field_name in (
            "canonical_kind",
            "memory_type",
            "slot_id",
            "canonical_subject",
            "identity_algorithm_version",
        ):
            if current_metadata.get(field_name) != desired_metadata.get(field_name):
                raise ValueError(f"canonical UPDATE cannot change immutable boundary: {field_name}")
        current_scope = dict(current_metadata.get("scope", {}) or {})
        desired_scope = dict(desired_metadata.get("scope", {}) or {})
        boundary_fields = ("canonical_subject", "applicability", "visibility")
        if canonical_json({key: current_scope.get(key) for key in boundary_fields}) != canonical_json(
            {key: desired_scope.get(key) for key in boundary_fields}
        ):
            raise ValueError("canonical UPDATE cannot weaken or change its scope")
        current_authority = dict(current_scope.get("authority", {}) or {})
        desired_authority = dict(desired_scope.get("authority", {}) or {})
        current_principals = {str(item) for item in current_authority.get("principal_ids", []) or []}
        desired_principals = {str(item) for item in desired_authority.get("principal_ids", []) or []}
        current_services = {str(item) for item in current_authority.get("service_ids", []) or []}
        desired_services = {str(item) for item in desired_authority.get("service_ids", []) or []}
        if (
            bool(desired_authority.get("inferred", False))
            or (current_principals and not desired_principals.issubset(current_principals))
            or (current_services and not desired_services.issubset(current_services))
        ):
            raise ValueError("canonical UPDATE cannot weaken or broaden assertion authority")

    def _preflight_regular_operations(
        self,
        operations: list[ContextOperation],
        *,
        validate_resolution_links: bool = True,
        validate_target_state: bool = True,
    ) -> None:
        """Parse every deterministic effect before any regular write occurs."""

        for operation in operations:
            if operation.payload.get("canonical_memory") is True or operation.status == OperationStatus.PENDING:
                continue
            marker = self._operation_marker(operation.operation_id)
            if marker.exists():
                self._validate_operation_marker(marker, operation)
                continue
            trusted_inflight = self._trusted_inflight_regular_object_effect(operation)
            self._validate_regular_operation_effect(
                trusted_inflight or operation,
                validate_target_state=validate_target_state,
                allow_existing_add=trusted_inflight is not None,
            )
            if trusted_inflight is not None:
                continue
            self._validate_pending_lifecycle_cas(
                operation,
                validate_resolution_links=validate_resolution_links,
            )

    def _validate_regular_operation_effect(
        self,
        operation: ContextOperation,
        *,
        validate_target_state: bool,
        allow_existing_add: bool = False,
    ) -> None:
        if not isinstance(operation.payload, dict):
            raise ValueError("regular operation payload must be an object")
        # All records written to redo, audit, diff, and the idempotency marker
        # must be serializable before the first SourceStore mutation.
        canonical_json(operation.to_dict())
        object_actions = {OperationAction.ADD, OperationAction.UPDATE, OperationAction.MERGE}
        desired_obj: ContextObject | None = None
        if operation.action in object_actions:
            object_payload = operation.payload.get("context_object")
            if not isinstance(object_payload, dict):
                raise ValueError(f"{operation.action.value} operation requires context_object")
            try:
                desired_obj = ContextObject.from_dict(object_payload)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("regular operation context_object is invalid") from exc
            if desired_obj.context_type != operation.context_type:
                raise ValueError("regular operation context_object type mismatch")
            if operation.target_uri and desired_obj.uri != operation.target_uri:
                raise ValueError("regular operation target_uri does not match context_object URI")
        elif operation.action == OperationAction.SUPERSEDE:
            object_payload = operation.payload.get("context_object")
            if not isinstance(object_payload, dict):
                raise ValueError("supersede operation requires context_object")
            try:
                desired_obj = ContextObject.from_dict(object_payload)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("supersede replacement context_object is invalid") from exc
            if desired_obj.context_type != operation.context_type:
                raise ValueError("supersede replacement context type mismatch")
            if validate_target_state and not operation.target_uri:
                raise ValueError("supersede operation requires an existing target URI")

        policy_actions = {
            OperationAction.REWARD,
            OperationAction.PENALIZE,
            OperationAction.COOLDOWN,
            OperationAction.SUPPRESS,
            OperationAction.DISABLE,
        }
        if operation.action == OperationAction.REWARD:
            RewardSignal.from_payload(operation.payload)
        elif operation.action == OperationAction.PENALIZE:
            PenaltySignal.from_payload(operation.payload)
        elif operation.action == OperationAction.COOLDOWN:
            cooldown_until = operation.payload.get("cooldown_until")
            if cooldown_until is not None and not isinstance(cooldown_until, str):
                raise ValueError("cooldown_until must be a string or null")

        target_actions = {
            OperationAction.UPDATE,
            OperationAction.MERGE,
            OperationAction.DELETE,
            OperationAction.ARCHIVE,
            OperationAction.COMPRESS,
            OperationAction.REFRESH_LAYERS,
            OperationAction.REINDEX,
            OperationAction.SUPERSEDE,
            *policy_actions,
        }
        current_target: ContextObject | None = None
        if operation.target_uri:
            try:
                current_target = self.source_store.read_object(operation.target_uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                current_target = None
        self._validate_regular_canonical_boundary(
            operation,
            current_target,
            desired_obj,
            allow_existing_add=allow_existing_add,
        )
        if validate_target_state and operation.action in target_actions:
            if not operation.target_uri:
                raise ValueError(f"{operation.action.value} operation requires a target URI")
            target = self.source_store.read_object(operation.target_uri)
            if target.context_type != operation.context_type:
                raise ValueError("regular operation target context type mismatch")
            if operation.action in policy_actions and operation.context_type == ContextType.ACTION_POLICY:
                self._read_action_policy(operation.target_uri)

    def _validate_regular_canonical_boundary(
        self,
        operation: ContextOperation,
        current: ContextObject | None,
        desired: ContextObject | None,
        *,
        allow_existing_add: bool,
    ) -> None:
        """Keep canonical Slot/Claim and pending objects on their formal paths."""

        def canonical_slot_or_claim(obj: ContextObject | None) -> bool:
            if obj is None:
                return False
            kind = str(dict(obj.metadata or {}).get("canonical_kind") or "")
            return (
                kind in {"slot", "claim"}
                or obj.schema_version == "canonical_memory_v2"
                or "/memories/canonical/" in obj.uri
            )

        def pending_proposal(obj: ContextObject | None) -> bool:
            if obj is None:
                return False
            metadata = dict(obj.metadata or {})
            return (
                metadata.get("canonical_kind") == "pending_proposal"
                or obj.schema_version == PendingMemoryProposal.SCHEMA_VERSION
                or "/memories/pending/" in obj.uri
            )

        if canonical_slot_or_claim(current) or canonical_slot_or_claim(desired):
            raise ValueError("canonical Slot and Claim mutations require a canonical transaction")

        current_pending = pending_proposal(current)
        desired_pending = pending_proposal(desired)
        lifecycle_transition = operation.payload.get("pending_lifecycle_transition") is True
        declares_pending = (
            operation.payload.get("canonical_pending_proposal") is True
            or lifecycle_transition
            or desired_pending
        )
        if operation.action == OperationAction.ADD:
            if lifecycle_transition:
                raise ValueError("pending proposal creation cannot declare a lifecycle transition")
            if declares_pending:
                if current is not None and not allow_existing_add:
                    raise ValueError("pending proposal ADD cannot overwrite an existing object")
                if desired is None or not desired_pending:
                    raise ValueError("pending proposal ADD requires a canonical pending object")
                pending = PendingMemoryProposal.from_context_object(desired)
                if (
                    pending.lifecycle_state != LifecycleState.PENDING
                    or pending.lifecycle_revision != 1
                    or pending.retry_count != 0
                    or pending.lifecycle_history
                ):
                    raise ValueError("pending proposal ADD must create the initial PENDING lifecycle revision")
                expected = PendingMemoryProposal.create(
                    pending.proposal,
                    pending.scope,
                    tenant_id=str(desired.tenant_id or "default"),
                    owner_user_id=str(desired.owner_user_id or ""),
                    source_role=pending.source_role,
                    pending_reason_code=pending.pending_reason_code,
                    request_identity=pending.request_identity,
                    related_existing_memory_ids=pending.related_existing_memory_ids,
                    retrieval_views=pending.retrieval_views,
                    created_at=pending.created_at,
                )
                expected_obj = pending.to_context_object(
                    tenant_id=str(desired.tenant_id or "default"),
                    owner_user_id=str(desired.owner_user_id or ""),
                )
                if (
                    operation.payload.get("canonical_pending_proposal") is not True
                    or desired.owner_user_id != operation.user_id
                    or operation.payload.get("tenant_id") != str(desired.tenant_id or "default")
                    or operation.payload.get("memory_type") != pending.proposal.memory_type
                    or pending.uri != expected.uri
                    or operation.target_uri != pending.uri
                    or operation.payload.get("content") != pending.content()
                    or operation.payload.get("pending_proposal_id") != pending.proposal_id
                    or operation.payload.get("schema_version") != PendingMemoryProposal.SCHEMA_VERSION
                    or canonical_json(desired.to_dict()) != canonical_json(expected_obj.to_dict())
                ):
                    raise ValueError("pending proposal ADD identity or content is invalid")
            return
        if current_pending:
            if operation.action != OperationAction.UPDATE or not lifecycle_transition or not desired_pending:
                raise ValueError("pending proposal mutations require a legal lifecycle UPDATE")
            return
        if declares_pending:
            raise ValueError("pending lifecycle flags cannot target a non-pending object")

    def _trusted_inflight_regular_object_effect(self, operation: ContextOperation) -> ContextOperation | None:
        if operation.action not in {OperationAction.ADD, OperationAction.UPDATE, OperationAction.MERGE}:
            return None
        matches = [
            entry
            for entry in self.redo.pending_entries()
            if entry.operation_id == operation.operation_id
            and entry.phase in {"source_written", "index_written", "audit_written", "diff_written"}
        ]
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError("regular operation has ambiguous redo state")
        persisted = matches[0].operation
        requested = operation
        if requested.target_uri is None and persisted.target_uri is not None:
            requested = ContextOperation.from_dict(operation.to_dict())
            requested.target_uri = persisted.target_uri
        if self._operation_effect_fingerprint(persisted) != self._operation_effect_fingerprint(requested):
            raise ValueError("regular redo operation conflicts with the requested effect")
        if not requested.target_uri:
            raise ValueError("regular redo operation is missing its persisted target")
        current = self.source_store.read_object(requested.target_uri)
        expected_payload = self._normalized_regular_object_effect(requested)
        if not isinstance(expected_payload, dict) or canonical_json(current.to_dict()) != canonical_json(expected_payload):
            raise ValueError("regular redo SourceStore effect does not match its operation")
        expected_content = str(requested.payload.get("content", ""))
        if expected_content or requested.action == OperationAction.ADD:
            try:
                actual_content = self.source_store.read_content(current.layers.l2_uri or current.uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                actual_content = ""
            if actual_content != expected_content:
                raise ValueError("regular redo SourceStore content does not match its operation")
        return requested

    def _validate_pending_lifecycle_cas(
        self,
        operation: ContextOperation,
        *,
        validate_resolution_links: bool = True,
    ) -> None:
        if operation.payload.get("pending_lifecycle_transition") is not True:
            return
        if operation.action != OperationAction.UPDATE or operation.context_type != ContextType.MEMORY:
            raise ValueError("pending lifecycle transition must be a memory UPDATE")
        target_uri = str(operation.target_uri or "")
        if not target_uri:
            raise ValueError("pending lifecycle transition requires a target URI")
        current_obj = self.source_store.read_object(target_uri)
        current = PendingMemoryProposal.from_context_object(current_obj)
        expected_state = str(operation.payload.get("expected_pending_lifecycle_state") or "")
        expected_revision = int(operation.payload.get("expected_pending_lifecycle_revision", 0) or 0)
        expected_updated_at = str(operation.payload.get("expected_pending_updated_at") or "")
        if (
            not expected_state
            or expected_revision < 1
            or not expected_updated_at
            or current.lifecycle_state.value != expected_state
            or current.lifecycle_revision != expected_revision
            or current.updated_at != expected_updated_at
        ):
            raise RevisionConflictError(
                "pending proposal lifecycle conflict: "
                f"expected {expected_state}@{expected_revision}, "
                f"actual {current.lifecycle_state.value}@{current.lifecycle_revision}"
            )
        desired_payload = operation.payload.get("context_object")
        if not isinstance(desired_payload, dict):
            raise ValueError("pending lifecycle transition requires context_object")
        desired_obj = ContextObject.from_dict(desired_payload)
        desired = PendingMemoryProposal.from_context_object(desired_obj)
        if (
            current_obj.uri != target_uri
            or current_obj.context_type != ContextType.MEMORY
            or current_obj.owner_user_id != operation.user_id
            or desired_obj.owner_user_id != current_obj.owner_user_id
            or str(desired_obj.tenant_id or "default") != str(current_obj.tenant_id or "default")
            or desired_obj.context_type != current_obj.context_type
        ):
            raise ValueError("pending lifecycle transition cannot change owner, tenant, URI, or context type")
        if current_obj.lifecycle_state != current.lifecycle_state or desired_obj.lifecycle_state != desired.lifecycle_state:
            raise ValueError("pending lifecycle object and payload state disagree")
        expected_current_obj = current.to_context_object(
            tenant_id=str(current_obj.tenant_id or "default"),
            owner_user_id=str(current_obj.owner_user_id or ""),
        )
        expected_desired_obj = desired.to_context_object(
            tenant_id=str(current_obj.tenant_id or "default"),
            owner_user_id=str(current_obj.owner_user_id or ""),
        )
        if canonical_json(current_obj.to_dict()) != canonical_json(expected_current_obj.to_dict()):
            raise ValueError("stored pending proposal object is internally inconsistent")
        if canonical_json(desired_obj.to_dict()) != canonical_json(expected_desired_obj.to_dict()):
            raise ValueError("pending lifecycle context_object is internally inconsistent")
        try:
            current_content = self.source_store.read_content(current_obj.layers.l2_uri or current_obj.uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as exc:
            raise ValueError("stored pending proposal content is missing") from exc
        if current_content != current.content():
            raise ValueError("stored pending proposal content does not match its object")
        if operation.payload.get("content") != desired.content():
            raise ValueError("pending lifecycle content does not match its desired object")
        if desired.uri != current.uri or desired.lifecycle_revision != current.lifecycle_revision + 1:
            raise ValueError("pending lifecycle transition must advance exactly one lifecycle revision")
        mutable_fields = {
            "lifecycle_state",
            "retry_count",
            "lifecycle_revision",
            "lifecycle_history",
            "updated_at",
        }
        current_core = {key: value for key, value in current.to_payload().items() if key not in mutable_fields}
        desired_core = {key: value for key, value in desired.to_payload().items() if key not in mutable_fields}
        if canonical_json(current_core) != canonical_json(desired_core):
            raise ValueError("pending lifecycle transition cannot rewrite proposal content or scope")

        retry_delta = desired.retry_count - current.retry_count
        if retry_delta not in {0, 1}:
            raise ValueError("pending lifecycle retry count must stay stable or increment once")
        if desired.lifecycle_state == current.lifecycle_state:
            if desired.lifecycle_state != LifecycleState.RETRYABLE or retry_delta != 1:
                raise ValueError("pending lifecycle transition cannot silently retain the current state")
        elif desired.lifecycle_state not in PENDING_PROPOSAL_TRANSITIONS.get(current.lifecycle_state, frozenset()):
            raise ValueError(
                "illegal pending proposal lifecycle transition: "
                f"{current.lifecycle_state.value}->{desired.lifecycle_state.value}"
            )
        if len(desired.lifecycle_history) != len(current.lifecycle_history) + 1 or canonical_json(
            desired.lifecycle_history[:-1]
        ) != canonical_json(current.lifecycle_history):
            raise ValueError("pending lifecycle history must append exactly one transition")
        expected_history = {
            "from": current.lifecycle_state.value,
            "to": desired.lifecycle_state.value,
            "from_revision": current.lifecycle_revision,
            "to_revision": desired.lifecycle_revision,
            "reason": str(operation.payload.get("pending_lifecycle_reason") or ""),
            "updated_at": desired.updated_at,
        }
        if canonical_json(desired.lifecycle_history[-1]) != canonical_json(expected_history):
            raise ValueError("pending lifecycle history does not match the requested transition")

        expected_fields = {
            "canonical_pending_proposal": True,
            "pending_proposal_id": desired.proposal_id,
            "pending_lifecycle_state": desired.lifecycle_state.value,
            "pending_lifecycle_revision": desired.lifecycle_revision,
            "memory_type": desired.proposal.memory_type,
            "schema_version": PendingMemoryProposal.SCHEMA_VERSION,
            "tenant_id": str(current_obj.tenant_id or "default"),
        }
        if any(operation.payload.get(key) != value for key, value in expected_fields.items()):
            raise ValueError("pending lifecycle operation envelope disagrees with its desired proposal")
        resolution_flag = operation.payload.get("pending_lifecycle_resolution")
        if not isinstance(resolution_flag, bool) or resolution_flag != (
            desired.lifecycle_state == LifecycleState.RESOLVED
        ):
            raise ValueError("pending lifecycle resolution flag disagrees with the desired state")
        resolution_keys = operation.payload.get("resolution_idempotency_keys", [])
        resolved_claims = operation.payload.get("resolved_claim_uris", [])
        if not isinstance(resolution_keys, list | tuple) or not isinstance(resolved_claims, list | tuple):
            raise ValueError("pending lifecycle resolution links must be lists")
        if not resolution_flag and (resolution_keys or resolved_claims):
            raise ValueError("non-RESOLVED pending transition cannot carry canonical resolution links")
        if resolution_flag and (not resolution_keys or not resolved_claims):
            raise ValueError("RESOLVED pending transition requires canonical resolution links")
        if resolution_flag and validate_resolution_links:
            self._validate_pending_resolution_commit(operation, current)

    def _validate_pending_resolution_commit(
        self,
        operation: ContextOperation,
        pending: PendingMemoryProposal,
    ) -> None:
        keys = tuple(
            dict.fromkeys(str(item) for item in operation.payload.get("resolution_idempotency_keys", []) or [] if item)
        )
        claim_uris = tuple(
            dict.fromkeys(str(item) for item in operation.payload.get("resolved_claim_uris", []) or [] if item)
        )
        if not keys or not claim_uris:
            raise ValueError("RESOLVED pending transition requires committed canonical Claim links")
        committed_claims_by_key: dict[str, set[str]] = {}
        for key in keys:
            marker = self._transaction_marker(key)
            if not marker.exists():
                raise RevisionConflictError("pending proposal cannot resolve before its canonical transaction commits")
            diff = self._transaction_marker_diff(marker)
            committed_claims_by_key[key] = {
                str(payload.get("uri"))
                for marker_operation in diff.operations
                if marker_operation.payload.get("idempotency_key") == key
                and isinstance((payload := marker_operation.payload.get("context_object")), dict)
                and dict(payload.get("metadata", {}) or {}).get("canonical_kind") == "claim"
            }
        operation_tenant = str(operation.payload.get("tenant_id") or "default")
        for uri in claim_uris:
            claim = self.source_store.read_object(uri)
            metadata = dict(claim.metadata or {})
            linked_key = str(metadata.get("canonical_idempotency_key") or "")
            if (
                claim.lifecycle_state != LifecycleState.ACTIVE
                or metadata.get("canonical_kind") != "claim"
                or metadata.get("state") != "ACTIVE"
                or claim.owner_user_id != operation.user_id
                or str(claim.tenant_id or "default") != operation_tenant
                or str(metadata.get("memory_type") or "") != pending.proposal.memory_type
                or linked_key not in keys
                or uri not in committed_claims_by_key.get(linked_key, set())
            ):
                raise RevisionConflictError("pending proposal resolution Claim is not the linked committed ACTIVE Claim")

    def _validate_pending_resolution_batch(self, operations: list[ContextOperation]) -> None:
        resolutions = [
            operation
            for operation in operations
            if operation.payload.get("canonical_pending_resolution") is True
        ]
        if not resolutions:
            return
        if len(resolutions) != 1:
            raise ValueError("canonical transaction can resolve exactly one pending proposal")
        resolution = resolutions[0]
        keys = {
            str(item)
            for item in resolution.payload.get("resolution_idempotency_keys", []) or []
            if item
        }
        transaction_keys = {
            str(operation.payload.get("idempotency_key") or "")
            for operation in operations
        }
        claim_uris = {
            str(item)
            for item in resolution.payload.get("resolved_claim_uris", []) or []
            if item
        }
        active_claims = {
            str(payload.get("uri") or ""): dict(payload.get("metadata", {}) or {})
            for operation in operations
            if operation is not resolution
            and isinstance((payload := operation.payload.get("context_object")), dict)
            and dict(payload.get("metadata", {}) or {}).get("canonical_kind") == "claim"
            and dict(payload.get("metadata", {}) or {}).get("state") == "ACTIVE"
        }
        if (
            len(transaction_keys) != 1
            or "" in transaction_keys
            or keys != transaction_keys
            or not claim_uris
            or not claim_uris.issubset(active_claims)
        ):
            raise ValueError("pending resolution must link ACTIVE Claims in the same canonical transaction")
        resolution_tenant = str(resolution.payload.get("tenant_id") or "default")
        resolution_memory_type = str(resolution.payload.get("memory_type") or "")
        for uri in claim_uris:
            claim_payload = next(
                operation.payload["context_object"]
                for operation in operations
                if isinstance(operation.payload.get("context_object"), dict)
                and str(operation.payload["context_object"].get("uri") or "") == uri
            )
            metadata = active_claims[uri]
            if (
                str(claim_payload.get("owner_user_id") or "") != resolution.user_id
                or str(claim_payload.get("tenant_id") or "default") != resolution_tenant
                or str(metadata.get("memory_type") or "") != resolution_memory_type
            ):
                raise ValueError("pending resolution Claim crosses owner, tenant, or memory type")

    def _finalize_canonical_outbox(
        self,
        transaction_id: str,
        idempotency_key: str,
        operations: list[ContextOperation],
        *,
        slot_uri: str | None = None,
    ) -> Path:
        outbox_path = self.root / "system" / "outbox" / f"{transaction_id}.json"
        outbox_complete = False
        if outbox_path.exists():
            try:
                existing = json.loads(outbox_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("canonical committed outbox is unreadable") from exc
            if existing.get("status") == "committed":
                existing_operations = [
                    ContextOperation.from_dict(item)
                    for item in existing.get("operations", []) or []
                    if isinstance(item, dict)
                ]
                if (
                    existing.get("transaction_id") != transaction_id
                    or existing.get("idempotency_key") != idempotency_key
                    or self._canonical_transaction_request_fingerprint(existing_operations)
                    != self._canonical_transaction_request_fingerprint(operations)
                    or self._canonical_transaction_effect_fingerprint(existing_operations)
                    != self._canonical_transaction_effect_fingerprint(operations)
                ):
                    raise ValueError("canonical committed outbox conflicts with its transaction marker")
                outbox_complete = True
        if not outbox_complete:
            outbox_path = self._write_outbox_event(
                transaction_id,
                idempotency_key,
                operations,
                status="committed",
            )
        resolved_slot = slot_uri or next(
            (
                str(payload.get("uri"))
                for operation in operations
                if isinstance((payload := operation.payload.get("context_object")), dict)
                and dict(payload.get("metadata", {}) or {}).get("canonical_kind") == "slot"
            ),
            transaction_id,
        )
        self._enqueue_outbox(transaction_id, resolved_slot, outbox_path, operations)
        return outbox_path

    def _preflight_canonical_revisions(
        self,
        operations: list[ContextOperation],
        *,
        check_revisions: bool = True,
    ) -> None:
        tenants: set[str] = set()
        owners: set[str] = set()
        slot_ids: set[str] = set()
        scope_payloads: set[str] = set()
        for operation in operations:
            object_payload = operation.payload.get("context_object")
            if not isinstance(object_payload, dict) or not object_payload.get("uri"):
                raise ValueError("canonical operation requires a context_object URI")
            uri = str(object_payload["uri"])
            metadata = dict(object_payload.get("metadata", {}) or {})
            if operation.payload.get("canonical_pending_resolution") is True:
                if (
                    object_payload.get("schema_version") != PendingMemoryProposal.SCHEMA_VERSION
                    or operation.payload.get("schema_version") != PendingMemoryProposal.SCHEMA_VERSION
                    or metadata.get("canonical_kind") != "pending_proposal"
                ):
                    raise ValueError("canonical pending resolution requires a pending proposal object")
                object_tenant = str(object_payload.get("tenant_id") or "default")
                operation_tenant = str(operation.payload.get("tenant_id") or "default")
                object_owner = str(object_payload.get("owner_user_id") or operation.user_id)
                if object_tenant != operation_tenant or object_owner != operation.user_id:
                    raise ValueError("canonical pending resolution tenant or owner mismatch")
                scope = dict(metadata.get("scope", {}) or {})
                subject_payload = scope.get("canonical_subject")
                if not isinstance(subject_payload, dict):
                    raise ValueError("canonical pending resolution requires an explicit subject")
                tenants.add(object_tenant)
                owners.add(object_owner)
                slot_ids.add(str(operation.payload.get("slot_id") or ""))
                scope_payloads.add(json.dumps(scope, ensure_ascii=False, sort_keys=True))
                if not operation.evidence or any(
                    not item.get("event_id") or not item.get("content_hash") for item in operation.evidence
                ):
                    raise ValueError("canonical pending resolution requires durable evidence references")
                self._validate_canonical_evidence(operation)
                if check_revisions:
                    self._validate_pending_lifecycle_cas(operation, validate_resolution_links=False)
                continue
            if object_payload.get("schema_version") != "canonical_memory_v2":
                raise ValueError("canonical operation requires canonical_memory_v2 object schema")
            if operation.payload.get("schema_version") != "canonical_memory_v2":
                raise ValueError("canonical operation requires canonical_memory_v2 transaction schema")
            if (
                metadata.get("identity_algorithm_version") != IDENTITY_ALGORITHM_V2
                or operation.payload.get("identity_algorithm_version") != IDENTITY_ALGORITHM_V2
            ):
                raise ValueError("canonical operation requires Identity V2")
            if "identity_alias_operations" in operation.payload:
                raise ValueError("Identity V2 canonical transactions cannot contain redirects")
            scope = dict(metadata.get("scope", {}) or {})
            subject_payload = scope.get("canonical_subject")
            subject_key = str(metadata.get("canonical_subject") or "")
            if not isinstance(subject_payload, dict) or not subject_key:
                raise ValueError("canonical operation requires an explicit canonical subject")
            if ScopeRef.from_dict(subject_payload).key != subject_key:
                raise ValueError("canonical operation subject payload does not match Identity V2")
            authority = dict(scope.get("authority", {}) or {})
            if not authority or bool(authority.get("inferred", False)):
                raise ValueError("canonical operation requires non-inferred assertion authority")
            object_tenant = str(object_payload.get("tenant_id") or "default")
            operation_tenant = str(operation.payload.get("tenant_id") or "default")
            object_owner = str(object_payload.get("owner_user_id") or operation.user_id)
            asserted_by = str(metadata.get("asserted_by") or operation.user_id)
            if (
                object_tenant != operation_tenant
                or object_owner != operation.user_id
                or asserted_by != operation.user_id
            ):
                raise ValueError("canonical operation tenant or owner does not match its transaction envelope")
            tenants.add(object_tenant)
            owners.add(object_owner)
            slot_ids.add(str(metadata.get("slot_id") or operation.payload.get("slot_id") or ""))
            scope_payloads.add(json.dumps(metadata.get("scope", {}), ensure_ascii=False, sort_keys=True))
            if not operation.evidence or any(
                not item.get("event_id") or not item.get("content_hash") for item in operation.evidence
            ):
                raise ValueError("canonical operation requires durable evidence references")
            self._validate_canonical_evidence(operation)
            if check_revisions:
                expected = int(operation.payload.get("expected_revision", 0))
                try:
                    current = self.source_store.read_object(uri)
                    actual = int(dict(current.metadata or {}).get("revision", 0))
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                    actual = 0
                if actual != expected:
                    raise RevisionConflictError(f"revision conflict for {uri}: expected {expected}, actual {actual}")
        if len(tenants) != 1 or len(slot_ids - {""}) != 1 or len(scope_payloads) != 1:
            raise ValueError("canonical transaction must preserve tenant, slot, and scope boundaries")
        self._validate_pending_resolution_batch(operations)

    def _validate_canonical_evidence(self, operation: ContextOperation) -> None:
        store = SessionArchiveStore(
            self.root,
            tenant_id=str(operation.payload.get("tenant_id") or "default"),
        )
        verified_sources: set[str] = set()
        operation_refs = {canonical_json(payload) for payload in operation.evidence}
        object_payload = operation.payload.get("context_object")
        if isinstance(object_payload, dict):
            metadata = dict(object_payload.get("metadata", {}) or {})
            for revision in metadata.get("revisions", []) or []:
                if not isinstance(revision, dict):
                    raise ValueError("canonical revision evidence payload must be an object")
                if int(revision.get("revision", 0)) != int(metadata.get("revision", 0)):
                    continue
                field_refs = dict(revision.get("field_evidence_refs", {}) or {})
                for field_name, refs in field_refs.items():
                    if not refs:
                        raise ValueError(f"canonical revision has no field evidence for {field_name}")
                    for ref in refs:
                        if canonical_json(ref) not in operation_refs:
                            raise ValueError(
                                f"canonical field evidence is missing from the transaction envelope: {field_name}"
                            )
        for payload in operation.evidence:
            source_uri = str(payload.get("source_uri") or "")
            if not source_uri:
                raise ValueError("canonical evidence requires a durable source_uri")
            if source_uri not in verified_sources:
                store.current_manifest(
                    source_uri,
                    tenant_id=str(operation.payload.get("tenant_id") or "default"),
                )
                verified_sources.add(source_uri)
            event_digest = str(payload.get("event_digest") or "")
            required = {
                "event_id",
                "event_digest",
                "event_schema_version",
                "tenant_id",
                "episode_id",
                "actor_id",
                "actor_kind",
                "actor_role",
                "actor_id_inferred",
                "actor_role_inferred",
                "subject_refs",
                "content_path",
                "occurred_at",
                "ingested_at",
                "sequence",
                "evidence_strength",
                "content_hash",
            }
            if any(name not in payload or payload[name] is None or payload[name] == "" for name in required):
                raise ValueError("canonical evidence reference is incomplete")
            event = store.read_event(
                source_uri,
                event_digest,
                tenant_id=str(operation.payload.get("tenant_id") or "default"),
            )
            if str(event.get("event_id")) != str(payload["event_id"]):
                raise ValueError("canonical evidence event ID does not match its immutable digest")
            if str(event.get("episode_id")) != str(payload["episode_id"]) or str(payload["episode_id"]) != str(
                operation.source_episode_id
            ):
                raise ValueError("canonical evidence event is not part of the source episode")
            if str(event.get("schema_version")) != str(payload["event_schema_version"]):
                raise ValueError("canonical evidence schema version mismatch")
            tenant_id = str(operation.payload.get("tenant_id") or "default")
            if str(event.get("tenant_id")) != str(payload["tenant_id"]) or str(payload["tenant_id"]) != tenant_id:
                raise ValueError("canonical evidence tenant mismatch")
            actor = dict(event.get("actor", {}) or {})
            for field_name, evidence_name in (
                ("id", "actor_id"),
                ("kind", "actor_kind"),
                ("role", "actor_role"),
                ("id_inferred", "actor_id_inferred"),
                ("role_inferred", "actor_role_inferred"),
            ):
                if actor.get(field_name) != payload[evidence_name]:
                    raise ValueError(f"canonical evidence actor mismatch: {evidence_name}")
            expected_subjects = tuple(canonical_json(item) for item in event.get("subjects", []) or [])
            if tuple(str(item) for item in payload.get("subject_refs", []) or []) != expected_subjects:
                raise ValueError("canonical evidence subject mismatch")
            content_path = str(payload["content_path"])
            if content_path != str(event.get("content_path") or ""):
                raise ValueError("canonical evidence content path mismatch")
            content = resolve_content_path(event.get("content"), content_path)
            text = content if isinstance(content, str) else canonical_json(content)
            if evidence_hash(text) != str(payload["content_hash"]):
                raise ValueError("canonical evidence content hash no longer matches the archive")
            if not self._same_evidence_time(event.get("occurred_at"), payload["occurred_at"]):
                raise ValueError("canonical evidence occurred_at mismatch")
            if not self._same_evidence_time(event.get("ingested_at"), payload["ingested_at"]):
                raise ValueError("canonical evidence ingested_at mismatch")
            if int(event.get("sequence", 0)) != int(payload["sequence"]):
                raise ValueError("canonical evidence sequence mismatch")
            inference = dict(event.get("inference", {}) or {})
            expected_strength = "INFERRED" if any(bool(value) for value in inference.values()) else "EXPLICIT"
            if str(payload["evidence_strength"]) != expected_strength:
                raise ValueError("canonical evidence strength mismatch")
            span_start = payload.get("span_start")
            span_end = payload.get("span_end")
            if (span_start is None) != (span_end is None):
                raise ValueError("canonical evidence span is incomplete")
            if span_start is None or span_end is None:
                continue
            start, end = int(span_start), int(span_end)
            if start < 0 or end <= start or end > len(text):
                raise ValueError("canonical evidence span is invalid")
            quoted_hash = payload.get("quoted_text_hash")
            quoted_text = text[start:end]
            if not quoted_hash or evidence_hash(quoted_text) != str(quoted_hash):
                raise ValueError("canonical evidence quote hash no longer matches the archive")
            if payload.get("quoted_text") != quoted_text:
                raise ValueError("canonical evidence quoted text no longer matches the archive")

    def _same_evidence_time(self, left: object, right: object) -> bool:
        from datetime import datetime, timezone

        try:
            left_time = datetime.fromisoformat(str(left).replace("Z", "+00:00"))
            right_time = datetime.fromisoformat(str(right).replace("Z", "+00:00"))
        except ValueError:
            return False
        if left_time.tzinfo is None:
            left_time = left_time.replace(tzinfo=timezone.utc)
        if right_time.tzinfo is None:
            right_time = right_time.replace(tzinfo=timezone.utc)
        return left_time.astimezone(timezone.utc) == right_time.astimezone(timezone.utc)

    def _validate_authoritative_batch(self, operations: list[ContextOperation]) -> None:
        slot_active: dict[str, str | None] = {}
        active_by_slot: dict[str, list[str]] = {}
        for operation in operations:
            payload = operation.payload.get("context_object")
            if not isinstance(payload, dict):
                continue
            metadata = dict(payload.get("metadata", {}) or {})
            if metadata.get("canonical_kind") == "slot":
                self._validate_existing_slot_invariant(str(payload.get("uri", "")))
                slot_active[str(metadata.get("slot_id", ""))] = (
                    str(metadata["active_claim_id"]) if metadata.get("active_claim_id") else None
                )
            elif (
                metadata.get("canonical_kind") == "claim"
                and metadata.get("transition_profile") == "AUTHORITATIVE_STATE"
                and metadata.get("state") == "ACTIVE"
            ):
                active_by_slot.setdefault(str(metadata.get("slot_id", "")), []).append(
                    str(metadata.get("claim_id", ""))
                )
        for slot_id, active_claims in active_by_slot.items():
            if len(active_claims) > 1:
                raise ValueError("authoritative slot cannot commit more than one ACTIVE claim")
            declared = slot_active.get(slot_id)
            if declared and active_claims and declared != active_claims[0]:
                raise ValueError("slot active_claim_id does not match active claim revision")

    def _validate_existing_slot_invariant(self, slot_uri: str) -> None:
        if not slot_uri:
            return
        try:
            slot = self.source_store.read_object(slot_uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return
        metadata = dict(slot.metadata or {})
        claim_ids = [str(item) for item in metadata.get("claim_ids", []) or []]
        active: list[str] = []
        for claim_id in claim_ids:
            try:
                claim = self.source_store.read_object(f"{slot_uri}/claims/{claim_id}")
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                continue
            claim_metadata = dict(claim.metadata or {})
            if str(claim_metadata.get("state", "")) == "ACTIVE":
                active.append(str(claim_metadata.get("claim_id", claim_id)))
        if len(active) > 1:
            raise ValueError(f"canonical slot invariant violation: multiple ACTIVE claims for {slot_uri}")
        pointer = str(metadata.get("active_claim_id") or "")
        if pointer and active and pointer != active[0]:
            raise ValueError(f"canonical slot invariant violation: active_claim_id mismatch for {slot_uri}")

    def _apply_canonical_source(self, operation: ContextOperation) -> None:
        payload = operation.payload.get("context_object")
        if not isinstance(payload, dict):
            raise ValueError("canonical operation requires context_object")
        obj = ContextObject.from_dict(payload)
        self.source_store.write_object(obj, content=str(operation.payload.get("content", "")))
        metadata = dict(obj.metadata or {})
        relation_metadata = {
            "tenant_id": obj.tenant_id or "default",
            "owner_user_id": obj.owner_user_id,
            "canonical_transaction_id": operation.payload.get("transaction_id"),
            "canonical_idempotency_key": operation.payload.get("idempotency_key"),
            "source_revision": metadata.get("revision"),
            "commit_group_id": operation.payload.get("commit_group_id"),
        }
        if self.relation_store is not None:
            for relation in obj.relations:
                self.relation_store.add_relation(
                    ContextRelation(
                        source_uri=relation.source_uri,
                        relation_type=relation.relation_type,
                        target_uri=relation.target_uri,
                        weight=relation.weight,
                        metadata={**dict(relation.metadata or {}), **relation_metadata},
                    )
                )
        if self.relation_store is not None and metadata.get("canonical_kind") == "claim":
            slot_uri = obj.uri.rsplit("/claims/", 1)[0]
            self._add_relation(obj.uri, "belongs_to_slot", slot_uri, relation_metadata)
            self._add_relation(slot_uri, "has_claim", obj.uri, relation_metadata)

    def _validate_existing_canonical_effect(self, operation: ContextOperation) -> None:
        payload = operation.payload.get("context_object")
        if not isinstance(payload, dict):
            raise ValueError("canonical operation requires context_object")
        desired = ContextObject.from_dict(payload)
        current = self.source_store.read_object(desired.uri)
        if canonical_json(current.to_dict()) != canonical_json(desired.to_dict()):
            raise RevisionConflictError(
                f"canonical recovery found a divergent object at desired revision: {desired.uri}"
            )
        expected_content = str(operation.payload.get("content", ""))
        try:
            actual_content = self.source_store.read_content(current.layers.l2_uri or current.uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            actual_content = ""
        if actual_content != expected_content:
            raise RevisionConflictError(
                f"canonical recovery found divergent content at desired revision: {desired.uri}"
            )

    def _write_outbox_event(
        self,
        transaction_id: str,
        idempotency_key: str,
        operations: list[ContextOperation],
        *,
        status: str = "committed",
        before_images: list[dict] | None = None,
    ) -> Path:
        path = self.root / "system" / "outbox" / f"{transaction_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        claim_revisions = []
        for operation in operations:
            payload = operation.payload.get("context_object")
            if not isinstance(payload, dict):
                continue
            metadata = dict(payload.get("metadata", {}) or {})
            if metadata.get("canonical_kind") == "claim":
                claim_revisions.append(
                    {
                        "uri": payload.get("uri"),
                        "claim_id": metadata.get("claim_id"),
                        "revision": metadata.get("revision"),
                    }
                )
        event: dict = {
            "event_type": "MemoryCommitted",
            "transaction_id": transaction_id,
            "idempotency_key": idempotency_key,
            "claim_revisions": claim_revisions,
            "operation_ids": [operation.operation_id for operation in operations],
            "operations": [operation.to_dict() for operation in operations],
            "status": status,
            "before_images": [self._before_image_payload(item) for item in (before_images or [])],
            "commit_group_id": next(
                (
                    str(operation.payload.get("commit_group_id"))
                    for operation in operations
                    if operation.payload.get("commit_group_id")
                ),
                "",
            ),
        }
        if path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            merged_claims = {
                str(item.get("uri")): item for item in existing.get("claim_revisions", []) or [] if item.get("uri")
            }
            for item in claim_revisions:
                current = merged_claims.get(str(item.get("uri")))
                if current is None or int(item.get("revision") or 0) >= int(current.get("revision") or 0):
                    merged_claims[str(item.get("uri"))] = item
            event["claim_revisions"] = list(merged_claims.values())
            event["operation_ids"] = list(
                dict.fromkeys(
                    [
                        *[str(item) for item in existing.get("operation_ids", []) or []],
                        *[operation.operation_id for operation in operations],
                    ]
                )
            )
            merged_operations = {
                str(item.get("operation_id")): item
                for item in existing.get("operations", []) or []
                if isinstance(item, dict) and item.get("operation_id")
            }
            for item in event["operations"]:
                if isinstance(item, dict) and item.get("operation_id"):
                    merged_operations[str(item["operation_id"])] = item
            event["operations"] = list(merged_operations.values())
            if not event["before_images"]:
                event["before_images"] = list(existing.get("before_images", []) or [])
        tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(event, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return path

    def _before_image_payload(self, snapshot: dict) -> dict:
        obj = snapshot.get("object")
        return {
            "uri": str(snapshot.get("uri", "")),
            "exists": bool(snapshot.get("exists")),
            "object": obj.to_dict() if isinstance(obj, ContextObject) else None,
            "content": str(snapshot.get("content", "")),
        }

    def _capture_canonical_state(self, operations: list[ContextOperation]) -> list[dict]:
        snapshots = []
        for operation in operations:
            payload = operation.payload.get("context_object")
            if not isinstance(payload, dict):
                continue
            uri = str(payload["uri"])
            try:
                obj = self.source_store.read_object(uri)
                exists = True
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                obj = None
                exists = False
            if obj is not None:
                try:
                    content = self.source_store.read_content(obj.layers.l2_uri or uri)
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                    content = ""
            else:
                content = ""
            relations = (
                self.relation_store.relations_of(
                    uri,
                    tenant_id=str(payload.get("tenant_id") or "default"),
                    owner_user_id=payload.get("owner_user_id"),
                )
                if self.relation_store is not None
                else []
            )
            snapshots.append({"uri": uri, "exists": exists, "object": obj, "content": content, "relations": relations})
        return snapshots

    def _restore_canonical_state(self, snapshots: list[dict]) -> None:
        for snapshot in reversed(snapshots):
            uri = str(snapshot["uri"])
            if snapshot["exists"]:
                self.source_store.write_object(snapshot["object"], content=str(snapshot["content"]))
                if snapshot["content"] == "":
                    obj = snapshot["object"]
                    self.source_store.write_content(obj.layers.l2_uri or uri, "")
            else:
                delete = getattr(self.source_store, "delete_object", None)
                if not callable(delete):
                    raise RuntimeError("SourceStore must support delete_object for canonical rollback")
                delete(uri)
            if self.relation_store is None:
                continue
            original = list(snapshot["relations"])
            current = self.relation_store.relations_of(uri)
            for relation in current:
                if relation not in original:
                    self.relation_store.delete_relation(
                        relation.source_uri,
                        relation.relation_type,
                        relation.target_uri,
                    )
            for relation in original:
                self.relation_store.add_relation(relation)

    def _enqueue_outbox(
        self,
        transaction_id: str,
        slot_uri: str,
        outbox_path: Path,
        operations: list[ContextOperation],
    ) -> None:
        if self.queue_store is None:
            return
        try:
            self.queue_store.enqueue(
                QueueJob(
                    job_id=f"outbox_{transaction_id}",
                    queue_name="memory_projection",
                    action="project_memory_committed",
                    target_uri=slot_uri,
                    payload={
                        "transaction_id": transaction_id,
                        "outbox_path": str(outbox_path),
                        "operation_ids": [operation.operation_id for operation in operations],
                    },
                )
            )
        except Exception as exc:
            self.audit.record(
                operations[0].user_id,
                "canonical_memory_outbox_enqueue_failed",
                {"transaction_id": transaction_id, "error_type": type(exc).__name__},
            )

    def _transaction_marker(self, idempotency_key: str) -> Path:
        return self.root / "system" / "transactions" / f"{idempotency_key}.json"

    def _write_transaction_marker(
        self,
        path: Path,
        diff: ContextDiff,
        operations: list[ContextOperation],
    ) -> None:
        if path.exists():
            self._validate_transaction_marker(path, operations)
            return
        payload = {
            "schema_version": "canonical_transaction_marker_v2",
            "request_fingerprint": self._canonical_transaction_request_fingerprint(operations),
            "effect_fingerprint": self._canonical_transaction_effect_fingerprint(operations),
            "diff": diff.to_dict(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def _validate_transaction_marker(
        self,
        path: Path,
        operations: list[ContextOperation],
    ) -> ContextDiff:
        diff = self._transaction_marker_diff(path)
        if (
            self._canonical_transaction_request_fingerprint(diff.operations)
            != self._canonical_transaction_request_fingerprint(operations)
            or self._canonical_transaction_effect_fingerprint(diff.operations)
            != self._canonical_transaction_effect_fingerprint(operations)
        ):
            raise ValueError("canonical idempotency marker conflicts with the requested transaction")
        return diff

    def _transaction_marker_diff(self, path: Path) -> ContextDiff:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "canonical_transaction_marker_v2":
            # Legacy transaction markers stored the full persisted diff. They
            # are still safe to compare because the desired operation payloads
            # are present in that record.
            return self._diff_from_payload(payload)
        diff_payload = payload.get("diff")
        if not isinstance(diff_payload, dict):
            raise ValueError("canonical transaction marker is missing its persisted diff")
        diff = self._diff_from_payload(diff_payload)
        if (
            payload.get("request_fingerprint")
            != self._canonical_transaction_request_fingerprint(diff.operations)
            or payload.get("effect_fingerprint")
            != self._canonical_transaction_effect_fingerprint(diff.operations)
        ):
            raise ValueError("canonical transaction marker integrity check failed")
        return diff

    def _canonical_transaction_request_fingerprint(self, operations: list[ContextOperation]) -> str:
        normalized = []
        for operation in sorted(operations, key=lambda item: item.operation_id):
            payload = operation.to_dict()
            payload.pop("status", None)
            normalized.append(payload)
        canonical_json(normalized)
        return stable_hash(normalized, length=64)

    def _canonical_transaction_effect_fingerprint(self, operations: list[ContextOperation]) -> str:
        effects = []
        for operation in sorted(operations, key=lambda item: item.operation_id):
            effects.append(
                {
                    "operation_id": operation.operation_id,
                    "user_id": operation.user_id,
                    "context_type": operation.context_type.value,
                    "action": operation.action.value,
                    "target_uri": operation.target_uri,
                    "context_object": operation.payload.get("context_object"),
                    "content": operation.payload.get("content", ""),
                }
            )
        canonical_json(effects)
        return stable_hash(effects, length=64)

    def _diff_from_payload(self, payload: dict) -> ContextDiff:
        return ContextDiff(
            user_id=str(payload["user_id"]),
            operations=[ContextOperation.from_dict(item) for item in payload.get("operations", [])],
            pending_operations=[ContextOperation.from_dict(item) for item in payload.get("pending_operations", [])],
            rejected_operations=[ContextOperation.from_dict(item) for item in payload.get("rejected_operations", [])],
            diff_id=str(payload.get("diff_id", "")),
            created_at=str(payload.get("created_at", "")),
            schema_version=str(payload.get("schema_version", "context_diff_v1")),
        )

    def resume(self, user_id: str, operation: ContextOperation, phase: str) -> bool:
        """处理 resume 这一步。"""

        if phase in {"committed"}:
            if operation.payload.get("canonical_memory") is not True:
                self._write_operation_marker(operation)
            self.redo.commit(operation.operation_id)
            return False
        if phase in {"started", "begin"}:
            diff = self.commit(user_id, [operation])
            return any(op.operation_id == operation.operation_id for op in diff.operations)
        if operation.payload.get("canonical_memory") is True:
            return self._resume_canonical(user_id, operation, phase)
        if phase == "source_written":
            self._apply_index(operation)
            self.redo.advance(operation, phase="index_written")
            self.audit.record(user_id, "context_operation_committed", operation.to_dict())
            self.redo.advance(operation, phase="audit_written")
            self._write_recovery_diff(user_id, operation)
            self.redo.advance(operation, phase="diff_written")
            self._write_operation_marker(operation)
            self.redo.commit(operation.operation_id)
            return True
        if phase == "index_written":
            self.audit.record(user_id, "context_operation_committed", operation.to_dict())
            self.redo.advance(operation, phase="audit_written")
            self._write_recovery_diff(user_id, operation)
            self.redo.advance(operation, phase="diff_written")
            self._write_operation_marker(operation)
            self.redo.commit(operation.operation_id)
            return True
        if phase == "audit_written":
            self._write_recovery_diff(user_id, operation)
            self.redo.advance(operation, phase="diff_written")
            self._write_operation_marker(operation)
            self.redo.commit(operation.operation_id)
            return True
        if phase == "diff_written":
            self._write_operation_marker(operation)
            self.redo.commit(operation.operation_id)
            return True
        return False

    def _resume_canonical(self, user_id: str, operation: ContextOperation, phase: str) -> bool:
        if phase == "source_written":
            self.audit.record(user_id, "canonical_memory_operation_committed", operation.to_dict())
            self.redo.advance(operation, phase="audit_written")
        transaction_id = str(operation.payload.get("transaction_id", ""))
        idempotency_key = str(operation.payload.get("idempotency_key", ""))
        object_payload = operation.payload.get("context_object")
        slot_uri = operation.target_uri or transaction_id
        if isinstance(object_payload, dict):
            metadata = dict(object_payload.get("metadata", {}) or {})
            if metadata.get("canonical_kind") == "claim":
                slot_uri = str(object_payload.get("uri", slot_uri)).rsplit("/claims/", 1)[0]
        outbox_path = self._write_outbox_event(transaction_id, idempotency_key, [operation])
        self._enqueue_outbox(transaction_id, slot_uri, outbox_path, [operation])
        self._write_recovery_diff(user_id, operation)
        self.redo.advance(operation, phase="diff_written")
        self.redo.commit(operation.operation_id)
        return True

    def resume_canonical_batch(self, user_id: str, entries: list) -> list[str]:  # noqa: ANN001
        """从事务日志记录的阶段继续完成整批写入。"""

        operations = [entry.operation for entry in entries]
        if not operations:
            return []
        transaction_ids = {str(operation.payload.get("transaction_id", "")) for operation in operations}
        idempotency_keys = {str(operation.payload.get("idempotency_key", "")) for operation in operations}
        if len(transaction_ids) != 1 or "" in transaction_ids or len(idempotency_keys) != 1:
            raise ValueError("canonical recovery requires one complete transaction")
        transaction_id = next(iter(transaction_ids))
        idempotency_key = next(iter(idempotency_keys))
        outbox_path = self.root / "system" / "outbox" / f"{transaction_id}.json"
        prepared = json.loads(outbox_path.read_text(encoding="utf-8"))
        expected_operation_ids = [str(item) for item in prepared.get("operation_ids", []) or []]
        by_id = {operation.operation_id: operation for operation in operations}
        for payload in prepared.get("operations", []) or []:
            operation = ContextOperation.from_dict(payload)
            by_id.setdefault(operation.operation_id, operation)
        if set(expected_operation_ids) != set(by_id):
            raise RuntimeError("canonical recovery outbox is missing transaction operations")
        ordered = [by_id[operation_id] for operation_id in expected_operation_ids]
        self._validate_canonical_envelope(user_id, ordered)
        self._preflight_canonical_revisions(ordered, check_revisions=False)
        self._validate_authoritative_batch(ordered)
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
            *(
                str(operation.target_uri)
                for operation in ordered
                if operation.payload.get("canonical_pending_resolution") is True and operation.target_uri
            ),
        }
        with ExitStack() as locks:
            for lock_key in sorted(lock_keys):
                locks.enter_context(self.path_lock.acquire(lock_key))
            marker = self._transaction_marker(idempotency_key)
            if marker.exists():
                diff = self._validate_transaction_marker(marker, ordered)
                self._finalize_canonical_outbox(
                    transaction_id,
                    idempotency_key,
                    diff.operations,
                    slot_uri=slot_uri,
                )
                for operation in ordered:
                    self.redo.commit(operation.operation_id)
                return [operation.operation_id for operation in diff.operations]
            for operation in ordered:
                payload = operation.payload.get("context_object")
                if not isinstance(payload, dict):
                    raise ValueError("canonical recovery requires context_object")
                uri = str(payload["uri"])
                if operation.payload.get("canonical_pending_resolution") is True:
                    desired_obj = ContextObject.from_dict(payload)
                    try:
                        current = self.source_store.read_object(uri)
                    except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as exc:
                        raise RevisionConflictError(
                            "canonical recovery cannot find its pending resolution target"
                        ) from exc
                    if canonical_json(current.to_dict()) == canonical_json(desired_obj.to_dict()):
                        self._validate_existing_canonical_effect(operation)
                    else:
                        self._validate_pending_lifecycle_cas(operation, validate_resolution_links=False)
                        self._apply_canonical_source(operation)
                    self.audit.record(
                        user_id,
                        "canonical_memory_operation_applied_during_recovery",
                        operation.to_dict(),
                    )
                    operation.status = OperationStatus.COMMITTED
                    continue
                expected = int(operation.payload.get("expected_revision", 0))
                desired_revision = int(dict(payload.get("metadata", {}) or {}).get("revision", 0))
                try:
                    actual = int(dict(self.source_store.read_object(uri).metadata or {}).get("revision", 0))
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                    actual = 0
                if actual == expected:
                    self._apply_canonical_source(operation)
                elif actual == desired_revision:
                    self._validate_existing_canonical_effect(operation)
                else:
                    raise RevisionConflictError(
                        f"canonical recovery conflict for {uri}: expected {expected} or {desired_revision}, actual {actual}"
                    )
                self.audit.record(user_id, "canonical_memory_operation_applied_during_recovery", operation.to_dict())
                operation.status = OperationStatus.COMMITTED
            diff = ContextDiff(user_id=user_id, operations=ordered, diff_id=f"diff_{transaction_id}")
            self.diff_writer.write(diff)
            self._write_transaction_marker(marker, diff, ordered)
            self.audit.record(
                user_id,
                "canonical_memory_transaction_recovered",
                {"transaction_id": transaction_id, "operation_ids": [item.operation_id for item in ordered]},
            )
            self._finalize_canonical_outbox(
                transaction_id,
                idempotency_key,
                ordered,
                slot_uri=slot_uri,
            )
            for operation in ordered:
                self.redo.commit(operation.operation_id)
            return [operation.operation_id for operation in ordered]

    def recover_pending_canonical(self, user_id: str) -> list[str]:
        """恢复卡在准备阶段或源数据已写入阶段的记忆事务。"""

        grouped: dict[str, list] = {}
        for entry in self.redo.pending_entries():
            if entry.operation.payload.get("canonical_memory") is not True:
                continue
            transaction_id = str(entry.operation.payload.get("transaction_id", ""))
            grouped.setdefault(transaction_id, []).append(entry)
        recovered = []
        for entries in grouped.values():
            recovered.extend(self.resume_canonical_batch(user_id, entries))
        return recovered

    def _write_recovery_diff(self, user_id: str, operation: ContextOperation) -> None:
        operation.status = OperationStatus.COMMITTED
        self.diff_writer.write(
            ContextDiff(user_id=user_id, operations=[operation], diff_id=f"diff_{operation.operation_id}")
        )

    def _operation_marker(self, operation_id: str) -> Path:
        return self.root / "system" / "operations" / f"{operation_id}.json"

    def _write_operation_marker(self, operation: ContextOperation) -> None:
        if operation.payload.get("canonical_memory") is True:
            return
        path = self._operation_marker(operation.operation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        stored_operation = operation.to_dict()
        stored_operation["status"] = OperationStatus.COMMITTED.value
        payload = {
            "schema_version": "operation_idempotency_marker_v2",
            "operation_id": operation.operation_id,
            "action": operation.action.value,
            "context_type": operation.context_type.value,
            "target_uri": operation.target_uri,
            "commit_group_id": operation.payload.get("commit_group_id"),
            "commit_consumer": operation.payload.get("commit_consumer"),
            "effect_fingerprint": self._operation_effect_fingerprint(operation),
            "operation": stored_operation,
            "status": "committed",
        }
        if path.exists():
            self._validate_operation_marker(path, operation)
            return
        tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        with tmp.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            self._validate_operation_marker(path, operation)
        finally:
            tmp.unlink(missing_ok=True)

    def _validate_operation_marker(self, path: Path, operation: ContextOperation) -> ContextOperation:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "operation_idempotency_marker_v2":
            raise ValueError("legacy operation marker cannot verify the persisted effect")
        expected = {
            "operation_id": operation.operation_id,
            "action": operation.action.value,
            "context_type": operation.context_type.value,
            "commit_group_id": operation.payload.get("commit_group_id"),
            "commit_consumer": operation.payload.get("commit_consumer"),
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("operation idempotency marker conflicts with the requested operation")
        stored_payload = payload.get("operation")
        if not isinstance(stored_payload, dict):
            raise ValueError("operation idempotency marker is missing its persisted operation")
        stored = ContextOperation.from_dict(stored_payload)
        if operation.target_uri not in {None, stored.target_uri} or payload.get("target_uri") != stored.target_uri:
            raise ValueError("operation idempotency marker conflicts with the requested target")
        requested = operation
        if requested.target_uri is None and stored.target_uri is not None:
            requested = ContextOperation.from_dict(operation.to_dict())
            requested.target_uri = stored.target_uri
        if (
            payload.get("effect_fingerprint") != self._operation_effect_fingerprint(stored)
            or payload.get("effect_fingerprint") != self._operation_effect_fingerprint(requested)
        ):
            raise ValueError("operation idempotency marker conflicts with the requested effect")
        stored.status = OperationStatus.COMMITTED
        return stored

    def _operation_effect_fingerprint(self, operation: ContextOperation) -> str:
        if operation.action in {OperationAction.ADD, OperationAction.UPDATE, OperationAction.MERGE}:
            effect_payload = {
                "context_object": self._normalized_regular_object_effect(operation),
                "content": operation.payload.get("content", ""),
            }
        elif operation.action == OperationAction.SUPERSEDE:
            effect_payload = {
                "context_object": self._normalized_regular_object_effect(operation),
                "content": operation.payload.get("content", ""),
                "reason": operation.payload.get("reason", operation.payload.get("supersede_reason", "")),
            }
        else:
            effect_payload = {
                key: value
                for key, value in operation.payload.items()
                if key not in {"target_resolution_reason", "target_candidates"}
            }
        effect = {
            "operation_id": operation.operation_id,
            "user_id": operation.user_id,
            "context_type": operation.context_type.value,
            "action": operation.action.value,
            "target_uri": operation.target_uri,
            "effect_payload": effect_payload,
        }
        canonical_json(effect)
        return stable_hash(effect, length=64)

    def _normalized_regular_object_effect(self, operation: ContextOperation) -> object:
        payload = operation.payload.get("context_object")
        if not isinstance(payload, dict):
            return payload
        obj = ContextObject.from_dict(payload)
        if operation.payload.get("content"):
            obj.layers = ContextLayers(
                l0_uri=f"{obj.uri}/.abstract.md",
                l1_uri=f"{obj.uri}/.overview.md",
                l2_uri=f"{obj.uri}/content.md",
            )
        return obj.to_dict()

    def _coalesce_non_policy_operations(self, operations: list[ContextOperation]) -> list[ContextOperation]:
        policy_actions = {
            OperationAction.REWARD,
            OperationAction.PENALIZE,
            OperationAction.COOLDOWN,
            OperationAction.SUPPRESS,
            OperationAction.DISABLE,
        }
        policy_ops = [operation for operation in operations if operation.action in policy_actions]
        other_ops = [operation for operation in operations if operation.action not in policy_actions]
        return [*self.coalescer.coalesce(other_ops), *policy_ops]

    def _apply_source(self, operation: ContextOperation) -> None:
        if operation.action == OperationAction.SUPERSEDE:
            self._apply_supersede_source(operation)
            return
        if operation.action in {OperationAction.ADD, OperationAction.UPDATE, OperationAction.MERGE}:
            object_payload = operation.payload.get("context_object")
            if isinstance(object_payload, dict):
                obj = ContextObject.from_dict(object_payload)
                content = str(operation.payload.get("content", ""))
                self.source_store.write_object(obj, content=content)
                if content:
                    LayerRefresher(self.source_store).refresh(obj, content)
                    operation.payload["context_object"] = obj.to_dict()
                self._apply_relations(obj, operation)
            return
        if (
            operation.action
            in {
                OperationAction.REWARD,
                OperationAction.PENALIZE,
                OperationAction.COOLDOWN,
                OperationAction.SUPPRESS,
                OperationAction.DISABLE,
            }
            and operation.target_uri
        ):
            if operation.context_type == ContextType.ACTION_POLICY:
                policy = self._read_action_policy(operation.target_uri)
                policy = self._apply_action_policy_mutation(policy, operation)
                self._write_action_policy(policy)
            elif operation.action == OperationAction.DISABLE:
                self.source_store.soft_delete(operation.target_uri, operation.action.value)
            return
        if operation.action == OperationAction.COMPRESS and operation.target_uri:
            obj = self.source_store.read_object(operation.target_uri)
            content = self._read_content_or_empty(operation.target_uri)
            LayerRefresher(self.source_store).refresh(
                obj, content, bullets=[operation.payload.get("reason", "compressed")]
            )
            obj.lifecycle_state = LifecycleState.COLD
            obj.metadata = {
                **obj.metadata,
                "compressed_at": utc_now(),
                "compression_reason": operation.payload.get("reason", ""),
            }
            self.source_store.write_object(obj)
            return
        if operation.action == OperationAction.REFRESH_LAYERS and operation.target_uri:
            obj = self.source_store.read_object(operation.target_uri)
            content = self._read_content_or_empty(operation.target_uri)
            LayerRefresher(self.source_store).refresh(obj, content)
            return
        if operation.action == OperationAction.ARCHIVE and operation.target_uri:
            obj = self.source_store.read_object(operation.target_uri)
            obj.lifecycle_state = LifecycleState.ARCHIVED
            obj.metadata = {
                **obj.metadata,
                "archived_at": utc_now(),
                "archive_reason": operation.payload.get("reason", ""),
            }
            content = self._read_content_or_empty(operation.target_uri)
            self.source_store.write_object(obj, content=content)
            return
        if operation.action == OperationAction.DELETE and operation.target_uri:
            self.source_store.soft_delete(operation.target_uri, operation.action.value)
            return

    def _apply_index(self, operation: ContextOperation) -> None:
        if operation.action == OperationAction.SUPERSEDE:
            self._apply_supersede_index(operation)
            return
        if operation.action in {OperationAction.ADD, OperationAction.UPDATE, OperationAction.MERGE}:
            object_payload = operation.payload.get("context_object")
            if isinstance(object_payload, dict):
                obj = ContextObject.from_dict(object_payload)
                self.index_store.upsert_index(obj, content=str(operation.payload.get("content", "")))
            return
        if operation.action == OperationAction.DELETE and operation.target_uri:
            self.index_store.delete_index(operation.target_uri)
            return
        if operation.target_uri and operation.action in {
            OperationAction.REWARD,
            OperationAction.PENALIZE,
            OperationAction.COOLDOWN,
            OperationAction.SUPPRESS,
            OperationAction.DISABLE,
            OperationAction.COMPRESS,
            OperationAction.REFRESH_LAYERS,
            OperationAction.ARCHIVE,
            OperationAction.REINDEX,
        }:
            if operation.action == OperationAction.DISABLE and operation.context_type != ContextType.ACTION_POLICY:
                self.index_store.delete_index(operation.target_uri)
                return
            obj = self.source_store.read_object(operation.target_uri)
            self.index_store.upsert_index(obj, content=self._read_content_or_empty(operation.target_uri))

    def _apply_action_policy_mutation(self, policy: ActionPolicy, operation: ContextOperation) -> ActionPolicy:
        if operation.action == OperationAction.REWARD:
            return self.action_policy_updater.reward(
                policy, RewardSignal.from_payload(operation.payload), operation_id=operation.operation_id
            )
        if operation.action == OperationAction.PENALIZE:
            return self.action_policy_updater.penalize(
                policy, PenaltySignal.from_payload(operation.payload), operation_id=operation.operation_id
            )
        if operation.action == OperationAction.COOLDOWN:
            return self.action_policy_updater.cooldown(
                policy, operation.payload.get("cooldown_until"), operation_id=operation.operation_id
            )
        if operation.action == OperationAction.SUPPRESS:
            return self.action_policy_updater.suppress(policy, operation_id=operation.operation_id)
        if operation.action == OperationAction.DISABLE:
            return self.action_policy_updater.disable_auto_execute(policy, operation_id=operation.operation_id)
        return policy

    def _apply_supersede_source(self, operation: ContextOperation) -> None:
        if not operation.target_uri:
            return
        object_payload = operation.payload.get("context_object")
        if not isinstance(object_payload, dict):
            return
        old_obj = self.source_store.read_object(operation.target_uri)
        old_content = self._read_content_or_empty(operation.target_uri)
        new_obj = ContextObject.from_dict(object_payload)
        new_obj.lifecycle_state = LifecycleState.ACTIVE
        superseded_at = utc_now()
        reason = str(operation.payload.get("reason") or operation.payload.get("supersede_reason") or "")
        old_obj.lifecycle_state = LifecycleState.OBSOLETE
        old_obj.metadata = {
            **old_obj.metadata,
            "superseded_at": superseded_at,
            "superseded_by": new_obj.uri,
            "supersede_reason": reason,
        }
        new_obj.metadata = {
            **new_obj.metadata,
            "supersedes": old_obj.uri,
            "superseded_at": superseded_at,
            "supersede_reason": reason,
        }
        self.source_store.write_object(old_obj, content=old_content)
        self.source_store.write_object(new_obj, content=str(operation.payload.get("content", "")))
        self._apply_relations(new_obj, operation)
        self._add_supersede_relations(old_obj, new_obj)

    def _apply_supersede_index(self, operation: ContextOperation) -> None:
        if not operation.target_uri:
            return
        old_obj = self.source_store.read_object(operation.target_uri)
        self.index_store.upsert_index(old_obj, content=self._read_content_or_empty(operation.target_uri))
        object_payload = operation.payload.get("context_object")
        if isinstance(object_payload, dict):
            new_uri = object_payload.get("uri")
            if not new_uri:
                return
            new_obj = self.source_store.read_object(str(new_uri))
            self.index_store.upsert_index(new_obj, content=str(operation.payload.get("content", "")))

    def _add_supersede_relations(self, old_obj: ContextObject, new_obj: ContextObject) -> None:
        metadata = {
            "tenant_id": new_obj.tenant_id or old_obj.tenant_id or "default",
            "owner_user_id": new_obj.owner_user_id or old_obj.owner_user_id,
        }
        self._add_relation(new_obj.uri, "supersedes", old_obj.uri, metadata)
        self._add_relation(old_obj.uri, "superseded_by", new_obj.uri, metadata)

    def _read_action_policy(self, uri: str) -> ActionPolicy:
        obj = self.source_store.read_object(uri)
        data = dict(obj.metadata)
        if not data:
            content = self._read_content_or_empty(uri)
            data = json.loads(content) if content else {}
        return ActionPolicy(**data)

    def _write_action_policy(self, policy: ActionPolicy) -> None:
        obj = policy.to_context_object()
        self.source_store.write_object(
            obj,
            content=json.dumps(policy.to_dict(), ensure_ascii=False, indent=2),
        )
        self._apply_relations(
            obj,
            ContextOperation(
                user_id=policy.user_id,
                context_type=ContextType.ACTION_POLICY,
                action=OperationAction.UPDATE,
                target_uri=policy.uri,
                payload={},
            ),
        )

    def _apply_relations(self, obj: ContextObject, operation: ContextOperation) -> None:
        if self.relation_store is None:
            return
        metadata = dict(obj.metadata)
        relation_metadata = {"tenant_id": obj.tenant_id or "default", "owner_user_id": obj.owner_user_id}
        if obj.context_type == ContextType.ACTION_POLICY:
            self._add_relation(obj.uri, "anchored_by", str(metadata.get("memory_anchor_uri", "")), relation_metadata)
            for uri in metadata.get("required_resource_uris", []) or []:
                self._add_relation(obj.uri, "requires_resource", str(uri), relation_metadata)
            for uri in metadata.get("required_skill_uris", []) or []:
                self._add_relation(obj.uri, "requires_skill", str(uri), relation_metadata)
            for uri in metadata.get("supported_behavior_pattern_uris", []) or []:
                self._add_relation(obj.uri, "supported_by", str(uri), relation_metadata)
            for uri in metadata.get("constrained_by_memory_uris", []) or []:
                self._add_relation(obj.uri, "constrained_by", str(uri), relation_metadata)
        elif obj.context_type in {ContextType.BEHAVIOR_PATTERN, ContextType.BEHAVIOR_CLUSTER}:
            self._add_relation(obj.uri, "anchored_by", str(metadata.get("memory_anchor_uri", "")), relation_metadata)
            for uri in metadata.get("case_refs", []) or []:
                self._add_relation(obj.uri, "aggregated_from", str(uri), relation_metadata)
            for uri in metadata.get("related_policy_uris", []) or metadata.get("policy_uris", []) or []:
                self._add_relation(str(uri), "supported_by", obj.uri, relation_metadata)
        elif obj.context_type == ContextType.MEMORY:
            for policy_uri in metadata.get("constrains_policy_uris", []) or []:
                self._add_relation(str(policy_uri), "constrained_by", obj.uri, relation_metadata)
            for behavior_uri in metadata.get("supporting_behavior_uris", []) or []:
                self._add_relation(obj.uri, "evidence_for", str(behavior_uri), relation_metadata)
        for relation in obj.relations:
            if self.relation_store is not None:
                self.relation_store.add_relation(relation)

    def _add_relation(self, source_uri: str, relation_type: str, target_uri: str, metadata: dict) -> None:
        if self.relation_store is None or not target_uri:
            return
        self.relation_store.add_relation(
            ContextRelation(
                source_uri=source_uri,
                relation_type=relation_type,
                target_uri=target_uri,
                metadata={key: value for key, value in metadata.items() if value is not None},
            )
        )

    def _read_content_or_empty(self, uri: str) -> str:
        try:
            return self.source_store.read_content(uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return ""
