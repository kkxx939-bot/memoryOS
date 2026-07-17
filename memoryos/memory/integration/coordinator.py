"""Canonical-memory coordination owned by the memory domain."""

from __future__ import annotations

from contextlib import ExitStack

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.contextdb.store.lock_store import LockLostError
from memoryos.contextdb.transaction.path_lock import LeaseGuard
from memoryos.core.errors import RevisionConflictError
from memoryos.core.ids import stable_hash
from memoryos.core.integrity import canonical_json
from memoryos.memory.canonical.current_head import (
    publish_current_head_sets,
)
from memoryos.memory.canonical.identity import canonical_text
from memoryos.memory.canonical.proposal import (
    PendingMemoryProposal,
)
from memoryos.memory.canonical.scope import ScopeRef
from memoryos.memory.canonical.visibility import (
    read_committed_canonical,
)
from memoryos.memory.integration.classification import is_canonical_memory_object
from memoryos.operations.commit.receipt import (
    load_transaction_receipt,
)
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus


class CanonicalCommitCoordinator:
    """Coordinate one canonical transaction without owning generic commit flow."""

    @staticmethod
    def _commit_canonical_batch(committer, user_id: str, operations: list[ContextOperation]) -> ContextDiff:
        if not operations:
            return ContextDiff(user_id=user_id)
        committer._validate_canonical_envelope(user_id, operations)
        transaction_ids = {str(operation.payload.get("transaction_id", "")) for operation in operations}
        idempotency_keys = {str(operation.payload.get("idempotency_key", "")) for operation in operations}
        if len(transaction_ids) != 1 or "" in transaction_ids or len(idempotency_keys) != 1 or "" in idempotency_keys:
            raise ValueError("canonical batch requires one transaction_id and idempotency_key")
        transaction_id = next(iter(transaction_ids))
        idempotency_key = next(iter(idempotency_keys))
        completed = committer._transaction_marker(idempotency_key)
        committer._reject_control_symlink(completed, "canonical transaction receipt")
        pending_entries = [
            entry
            for entry in committer.redo.pending_entries()
            if str(entry.operation.payload.get("transaction_id") or "") == transaction_id
        ]
        if completed.exists() and pending_entries:
            committer.resume_canonical_batch(user_id, pending_entries)
            return committer._validate_transaction_marker(completed, operations)

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
            f"canonical-idempotency:{idempotency_key}",
            f"canonical-transaction:{transaction_id}",
            *(
                str(operation.target_uri)
                for operation in operations
                if committer._canonical_pending_effect(operation) and operation.target_uri
            ),
        }
        with ExitStack() as locks:
            guards: list[LeaseGuard] = []
            for lock_key in sorted(lock_keys):
                guards.append(locks.enter_context(committer.path_lock.acquire(committer._lock_key(lock_key))))
            with committer.path_lock.fenced(guards):
                if completed.exists():
                    diff = committer._validate_transaction_marker(completed, operations)
                    committer._ensure_canonical_planning_digest(operations)
                    receipt = load_transaction_receipt(completed)
                    committer._validate_head_published_receipt(completed, receipt)
                    committer._finalize_canonical_outbox(transaction_id, idempotency_key, diff.operations)
                    return diff
                committer._preflight_canonical_revisions(operations)
                committer._validate_authoritative_batch(operations)
                committer.final_state_validator.validate(
                    operations,
                    tenant_id=committer.tenant_id,
                    owner_user_id=user_id,
                )
                committer._ensure_canonical_planning_digest(operations)
                backups = committer._capture_canonical_state(operations)
                before_by_uri = {
                    str(snapshot["uri"]): (
                        snapshot["object"] if isinstance(snapshot.get("object"), ContextObject) else None
                    )
                    for snapshot in backups
                }
                relation_manifests = {
                    operation.operation_id: committer._build_canonical_relation_manifest(
                        operation,
                        before_by_uri.get(str(operation.target_uri or "")),
                    )
                    for operation in operations
                }
                committed: list[ContextOperation] = []
                committer._notify("before_redo", transaction_id)
                committer._write_outbox_event(
                    transaction_id,
                    idempotency_key,
                    operations,
                    status="prepared",
                    before_images=backups,
                    relation_manifests=relation_manifests,
                )
                for operation in operations:
                    committer.redo.begin(
                        operation,
                        phase="started",
                        relation_manifest=relation_manifests[operation.operation_id],
                    )
                committer._notify("after_redo_begin", transaction_id)
            try:
                for operation in operations:
                    with committer.path_lock.fenced(guards):
                        committer._apply_canonical_source(operation)
                        committer._notify("after_source_effect", transaction_id)
                        committer._apply_canonical_relation_manifest(
                            operation,
                            relation_manifests[operation.operation_id],
                        )
                        committer._notify("after_relation_effect", transaction_id)
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
                        committer.audit.record(user_id, "canonical_memory_operation_applied", operation.to_dict())
                        committer._notify("after_audit", transaction_id)
                        committer.redo.advance(operation, phase="audit_written")
                        operation.status = OperationStatus.COMMITTED
                        committed.append(operation)
                with committer.path_lock.fenced(guards):
                    committer._write_outbox_event(
                        transaction_id,
                        idempotency_key,
                        committed,
                        status="source_committed",
                        before_images=backups,
                        relation_manifests=relation_manifests,
                    )
            except LockLostError:
                raise
            except Exception:
                with committer.path_lock.fenced(guards):
                    committer._restore_canonical_state(backups)
                    committer._write_outbox_event(
                        transaction_id,
                        idempotency_key,
                        operations,
                        status="aborted",
                    )
                    for operation in operations:
                        committer.redo.commit(operation.operation_id)
                    committer.audit.record(
                        user_id,
                        "canonical_memory_transaction_rolled_back",
                        {"transaction_id": transaction_id, "operation_ids": [item.operation_id for item in operations]},
                    )
                raise
            with committer.path_lock.fenced(guards):
                diff = committer._ensure_canonical_transaction_diff(
                    user_id,
                    transaction_id,
                    committed,
                )
                committer._notify("after_diff", transaction_id)
                committer._notify("before_receipt", transaction_id)
                committer._write_transaction_marker(
                    completed,
                    diff,
                    committed,
                    relation_manifests=relation_manifests,
                )
                receipt = load_transaction_receipt(completed)
                committer._notify("after_receipt", transaction_id)
                committer._notify("before_current_head", transaction_id)
                publish_current_head_sets(committer.artifact_root, completed, receipt)
                committer._mark_current_heads_published(committed)
                committer._notify("after_current_head", transaction_id)
                committer.audit.record(
                    user_id,
                    "canonical_memory_transaction_committed",
                    {"transaction_id": transaction_id, "operation_ids": [item.operation_id for item in committed]},
                )
                committer._finalize_canonical_outbox(transaction_id, idempotency_key, committed, slot_uri=slot_uri)
                committer._notify("before_redo_cleanup", transaction_id)
                for operation in committed:
                    committer.redo.commit(operation.operation_id)
                return diff

    @staticmethod
    def _preflight_canonical_groups(
        committer,
        user_id: str,
        groups: list[list[ContextOperation]],
    ) -> None:
        """Validate every group before the first group can create a side effect."""

        virtual_revisions: dict[str, int] = {}
        idempotency_transactions: dict[str, str] = {}
        for operations in groups:
            if not operations:
                continue
            committer._validate_canonical_envelope(user_id, operations)
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
            committer._reject_control_symlink(
                committer.artifact_root / "system" / "diffs" / f"diff_{transaction_id}.json",
                "canonical diff artifact",
            )
            existing_transaction = idempotency_transactions.setdefault(idempotency_key, transaction_id)
            if existing_transaction != transaction_id:
                raise ValueError("canonical idempotency key cannot identify multiple transactions")
            committer._canonical_transaction_request_fingerprint(operations)
            committer._canonical_transaction_effect_fingerprint(operations)
            marker = committer._transaction_marker(idempotency_key)
            committer._reject_control_symlink(marker, "canonical transaction receipt")
            if marker.exists():
                planning_error: ValueError | None = None
                try:
                    committer._ensure_canonical_planning_digest(operations)
                except ValueError as exc:
                    planning_error = exc
                committer._validate_transaction_marker(marker, operations)
                if planning_error is not None:
                    raise planning_error
                continue
            for operation in operations:
                if committer._canonical_pending_effect(operation):
                    committer._validate_pending_lifecycle_cas(operation, validate_resolution_links=False)
            committer._validate_pending_resolution_batch(operations)
            committer._validate_pending_correction_batch(operations)
            committer._preflight_canonical_revisions(operations, check_revisions=False)
            committer._validate_authoritative_batch(operations)
            for operation in operations:
                if committer._canonical_pending_effect(operation):
                    continue
                object_payload = operation.payload.get("context_object")
                assert isinstance(object_payload, dict)
                uri = str(object_payload["uri"])
                if uri not in virtual_revisions:
                    try:
                        current = read_committed_canonical(
                            committer.source_store,
                            uri,
                            committer.relation_store,
                        ).object
                        virtual_revisions[uri] = int(dict(current.metadata or {}).get("revision", 0))
                    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                        virtual_revisions[uri] = 0
                expected = int(operation.payload.get("expected_revision", 0))
                if virtual_revisions[uri] != expected:
                    raise RevisionConflictError(
                        f"revision conflict for {uri}: expected {expected}, actual {virtual_revisions[uri]}"
                    )
                virtual_revisions[uri] = int(dict(object_payload.get("metadata", {}) or {}).get("revision", 0))
            # Preserve revision-CAS as the first conflict classification for a
            # stale but otherwise well-formed operation set.  The immutable
            # planning proof remains mandatory and is still verified before
            # any artifact is published or source effect is written.
            committer._ensure_canonical_planning_digest(operations, publish=False)

    @staticmethod
    def _validate_canonical_envelope(committer, user_id: str, operations: list[ContextOperation]) -> None:
        """Validate immutable ownership boundaries before any marker fast path."""

        committer._validate_and_bind_operations(user_id, operations)
        if not user_id:
            raise ValueError("canonical commit requires a user_id")
        for operation in operations:
            committer._validate_canonical_artifact_keys(operation)
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
            operation_tenant = str(operation.payload.get("tenant_id") or committer.tenant_id)
            object_tenant = str(obj.tenant_id or committer.tenant_id)
            if object_tenant != operation_tenant:
                raise ValueError("canonical context object tenant does not match operation tenant")
            if operation_tenant != committer.tenant_id:
                raise ValueError("canonical operation tenant does not match bound tenant")
            metadata = dict(obj.metadata or {})
            if committer._canonical_pending_effect(operation):
                if (
                    operation.action != OperationAction.UPDATE
                    or operation.payload.get("pending_lifecycle_transition") is not True
                    or metadata.get("canonical_kind") != "pending_proposal"
                    or obj.schema_version != PendingMemoryProposal.SCHEMA_VERSION
                ):
                    raise ValueError("canonical pending lifecycle envelope is invalid")
                is_resolution = operation.payload.get("canonical_pending_resolution") is True
                if is_resolution != (operation.payload.get("pending_lifecycle_resolution") is True):
                    raise ValueError("canonical pending lifecycle kind disagrees with its terminal state")
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
            if str(visibility.get("tenant_id") or operation_tenant) != operation_tenant:
                raise ValueError("canonical visibility scope crosses the operation tenant")
            principals = {str(item) for item in dict(scope.get("authority", {}) or {}).get("principal_ids", []) or []}
            if principals and user_id not in principals:
                raise ValueError("canonical assertion scope does not authorize the commit user")
            if int(operation.payload.get("expected_revision", 0) or 0) > 0:
                committer._validate_existing_canonical_boundary(obj)

    @staticmethod
    def _validate_existing_canonical_boundary(committer, desired: ContextObject) -> None:
        try:
            current = read_committed_canonical(
                committer.source_store,
                desired.uri,
                committer.relation_store,
            ).object
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

    @staticmethod
    def _reject_canonical_regular_bypass(committer, operations: list[ContextOperation]) -> None:
        for operation in operations:
            target = str(operation.target_uri or "")
            raw = operation.payload.get("context_object")
            metadata = dict(raw.get("metadata", {}) or {}) if isinstance(raw, dict) else {}
            kind = str(metadata.get("canonical_kind") or "")
            schema_version = str(raw.get("schema_version") or "") if isinstance(raw, dict) else ""
            payload_is_canonical = schema_version in {
                "canonical_memory_v2",
                PendingMemoryProposal.SCHEMA_VERSION,
            }
            existing_is_canonical = False
            existing_kind = ""
            if target:
                try:
                    existing = committer.source_store.read_object(target)
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                    existing = None
                if existing is not None and is_canonical_memory_object(existing):
                    existing_is_canonical = True
                    existing_kind = str(dict(existing.metadata or {}).get("canonical_kind") or "")
            effective_kind = kind or existing_kind
            pending_target = bool(
                effective_kind == "pending_proposal"
                or schema_version == PendingMemoryProposal.SCHEMA_VERSION
                or "/memories/pending/" in target
            )
            canonical_target = bool(
                effective_kind in {"slot", "claim"}
                or "/memories/canonical/" in target
                or (payload_is_canonical and not pending_target)
                or (existing_is_canonical and not pending_target)
            )
            if canonical_target:
                if operation.payload.get("canonical_memory") is not True:
                    raise ValueError(
                        "canonical Slot/Claim operations require a canonical transaction and final-state validator"
                    )
            if pending_target:
                if operation.payload.get("canonical_pending_proposal") is not True:
                    raise ValueError(
                        "pending-memory objects require a legal lifecycle UPDATE through committed lifecycle validation"
                    )
