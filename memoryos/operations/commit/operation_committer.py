"""Stable OperationCommitter facade.

Implementation is split by transaction responsibility.  Every delegated method
remains explicit so callers and fault-injection tests can replace the same
observable hooks without dynamic attribute forwarding.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.index_store import IndexStore
from memoryos.contextdb.store.lock_store import LockStore
from memoryos.contextdb.store.queue_store import QueueStore
from memoryos.contextdb.store.relation_store import RelationStore
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.contextdb.transaction.path_lock import LeaseGuard, PathLock
from memoryos.core.durable_io import atomic_create_json as atomic_create_json
from memoryos.operations.commit.audit_diff import CommitAuditDiff
from memoryos.operations.commit.audit_writer import AuditWriter
from memoryos.operations.commit.coordinator import CommitCoordinator
from memoryos.operations.commit.diff_writer import DiffWriter
from memoryos.operations.commit.domain_protocols import ActionPolicy, AliasRegistry, PendingMemoryProposal
from memoryos.operations.commit.domain_registry import (
    RegisteredActionPolicyCommitHandlers,
    RegisteredMemoryCommitHandlers,
    action_policy_commit_handlers,
    memory_commit_handlers,
)
from memoryos.operations.commit.effects.canonical import CanonicalEffectExecutor
from memoryos.operations.commit.effects.regular import RegularEffectExecutor
from memoryos.operations.commit.effects.writer import StoreEffectWriter
from memoryos.operations.commit.markers.operation import OperationMarkerStore
from memoryos.operations.commit.markers.transaction import TransactionMarkerStore
from memoryos.operations.commit.operation_coalescer import OperationCoalescer
from memoryos.operations.commit.ordinary_relation import (
    commit_ordinary_relation_update as execute_ordinary_relation_update,
)
from memoryos.operations.commit.outbox import CommitOutbox
from memoryos.operations.commit.planning_proof import (
    ImmutablePlanningProofStore,
)
from memoryos.operations.commit.recovery_state_machine import CommitRecoveryStateMachine
from memoryos.operations.commit.redo_log import RedoEntry, RedoLog
from memoryos.operations.commit.state_machine import CommitStateMachine
from memoryos.operations.commit.validation import RegularOperationValidator
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.resolver.conflict_resolver import ConflictResolver
from memoryos.operations.resolver.target_resolver import TargetResolver

if TYPE_CHECKING:
    from memoryos.contextdb.ordinary_relations import OrdinaryRelationEligibility


class OperationCommitter:
    """Coordinate durable commits while delegating each bounded responsibility."""

    @staticmethod
    def _canonical_pending_effect(operation: ContextOperation) -> bool:
        return (
            operation.payload.get("canonical_pending_resolution") is True
            or operation.payload.get("canonical_pending_correction") is True
        )

    @staticmethod
    def _atomic_create_json(path, payload, *, artifact_root):  # noqa: ANN001, ANN202
        """Preserve the facade's historical atomic-publication test hook."""

        return atomic_create_json(path, payload, artifact_root=artifact_root)

    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        root: str,
        lock_store: LockStore | None = None,
        relation_store: RelationStore | None = None,
        queue_store: QueueStore | None = None,
        target_resolver: TargetResolver | None = None,
        tenant_id: str | None = None,
        test_hook=None,  # noqa: ANN001
        alias_registry: AliasRegistry | None = None,
        tombstone_service=None,  # noqa: ANN001
        migration_gate=None,  # noqa: ANN001
    ) -> None:
        source_tenant = getattr(source_store, "tenant_id", None)
        if source_tenant is not None:
            source_tenant = self._validate_tenant_id(source_tenant, "SourceStore tenant_id")
        bound_tenant = (
            self._validate_tenant_id(tenant_id, "OperationCommitter tenant_id")
            if tenant_id is not None
            else source_tenant or "default"
        )
        if source_tenant is not None and source_tenant != bound_tenant:
            raise ValueError("OperationCommitter tenant does not match SourceStore tenant")
        self.source_store = source_store
        self.index_store = index_store
        self.relation_store = relation_store
        self.queue_store = queue_store
        self.root = Path(root)
        self.artifact_root = self.root if bound_tenant == "default" else self.root / "tenants" / bound_tenant
        self.tenant_id = bound_tenant
        self._memory_commit_handlers = memory_commit_handlers()
        self._action_policy_commit_handlers = action_policy_commit_handlers()
        self.final_state_validator: Any
        if self._memory_commit_handlers is None:
            self.domain_overlay = None
            self.relation_domain_policy = None
            self.final_state_validator = None
            self.planning_envelopes = None
        else:
            self.domain_overlay = self._memory_commit_handlers.domain_classifier_binder(
                source_store,
                index_store,
                relation_store,
            )
            self.relation_domain_policy = self._memory_commit_handlers.relation_domain_policy_factory()
            self.final_state_validator = self._memory_commit_handlers.final_state_validator_factory(
                source_store,
                relation_store,
                alias_registry,
            )
            self.planning_envelopes = self._memory_commit_handlers.planning_envelope_store_factory(
                self.root,
                self.tenant_id,
            )
        self.coalescer = OperationCoalescer()
        self.conflicts = ConflictResolver()
        self.target_resolver = target_resolver or TargetResolver(index_store, source_store=source_store)
        self.redo = RedoLog(self.artifact_root)
        self.diff_writer = DiffWriter(self.artifact_root)
        self.audit = AuditWriter(self.artifact_root)
        resolved_lock_store = lock_store
        if resolved_lock_store is None:
            lock_store_provider = getattr(source_store, "operation_lock_store", None)
            if callable(lock_store_provider):
                candidate = lock_store_provider()
                required_methods = ("acquire", "renew", "assert_owned", "fenced", "release")
                if not all(callable(getattr(candidate, method, None)) for method in required_methods):
                    raise TypeError("SourceStore operation_lock_store returned an invalid LockStore")
                resolved_lock_store = cast(LockStore, candidate)
        if resolved_lock_store is None:
            raise RuntimeError("OperationCommitter requires an injected LockStore")
        self.path_lock = PathLock(resolved_lock_store)
        self.action_policy_updater = (
            self._action_policy_commit_handlers.updater_factory()
            if self._action_policy_commit_handlers is not None
            else None
        )
        self.test_hook = test_hook
        self.tombstone_service = tombstone_service
        self.migration_gate = migration_gate
        self.planning_proofs = ImmutablePlanningProofStore(self.artifact_root, tenant_id=self.tenant_id)
        self._startup_recovery_group: ContextVar[str] = ContextVar(
            f"memoryos_startup_recovery_group_{id(self)}",
            default="",
        )
        self._projection_fence_depth: ContextVar[int] = ContextVar(
            f"memoryos_operation_projection_fence_depth_{id(self)}",
            default=0,
        )

    def _require_memory_commit_handlers(self) -> RegisteredMemoryCommitHandlers:
        handlers = self._memory_commit_handlers
        if handlers is None:
            raise RuntimeError("Memory commit handlers are not registered")
        return handlers

    def _require_action_policy_commit_handlers(self) -> RegisteredActionPolicyCommitHandlers:
        handlers = self._action_policy_commit_handlers
        if handlers is None:
            raise RuntimeError("ActionPolicy commit handlers are not registered")
        return handlers

    def _load_canonical_current_head(self, uri: str):  # noqa: ANN201
        handlers = self._require_memory_commit_handlers()
        return handlers.load_current_head(self.artifact_root, uri)

    def _canonical_current_head_error(self) -> type[Exception]:
        return self._require_memory_commit_handlers().current_head_integrity_error

    def _publish_canonical_current_heads(self, marker: Path, receipt: dict) -> None:
        handlers = self._require_memory_commit_handlers()
        handlers.publish_current_head_sets(self.artifact_root, marker, receipt)

    def _read_committed_canonical(self, uri: str):  # noqa: ANN201
        handlers = self._require_memory_commit_handlers()
        return handlers.read_committed_canonical(self.source_store, uri, self.relation_store)

    def _committed_canonical_content(self, committed) -> str:  # noqa: ANN001
        return self._require_memory_commit_handlers().committed_content(committed)

    def _committed_canonical_relations(self, committed):  # noqa: ANN001, ANN201
        return self._require_memory_commit_handlers().committed_relations(committed)

    def _session_evidence_reader(self, tenant_id: str | None = None):  # noqa: ANN201
        handlers = self._require_memory_commit_handlers()
        return handlers.session_evidence_reader_factory(
            self.root,
            tenant_id or self.tenant_id,
        )

    @staticmethod
    def _delete_tombstone_ids(operation: ContextOperation) -> tuple[str, ...]:
        return CommitStateMachine._delete_tombstone_ids(operation)

    def _require_delete_tombstone_capability(self, operations: list[ContextOperation]) -> None:
        return CommitStateMachine._require_delete_tombstone_capability(self, operations)

    def _prepare_delete_tombstones(
        self, operation: ContextOperation, *, trust_durable_binding: bool = False
    ) -> tuple[str, ...]:
        return CommitStateMachine._prepare_delete_tombstones(
            self, operation, trust_durable_binding=trust_durable_binding
        )

    def _settle_delete_tombstones(self, operations: list[ContextOperation]) -> None:
        return CommitStateMachine._settle_delete_tombstones(self, operations)

    @contextmanager
    def _durable_startup_recovery_scope(self, group_id: str) -> Iterator[None]:
        with CommitStateMachine._durable_startup_recovery_scope(self, group_id):
            yield

    @contextmanager
    def _migration_projection_fence(self) -> Iterator[None]:
        with CommitStateMachine._migration_projection_fence(self):
            yield

    def _require_commit_ready(self, user_id: str, operations: list[ContextOperation]) -> None:
        return CommitStateMachine._require_commit_ready(self, user_id, operations)

    def _validate_durable_startup_commit(self, group_id: str, user_id: str, operations: list[ContextOperation]) -> None:
        return CommitStateMachine._validate_durable_startup_commit(self, group_id, user_id, operations)

    def _notify(self, stage: str, transaction_id: str) -> None:
        return CommitStateMachine._notify(self, stage, transaction_id)

    def _mark_current_heads_published(self, operations: list[ContextOperation]) -> None:
        return CommitStateMachine._mark_current_heads_published(self, operations)

    def _validate_head_published_receipt(self, receipt_path: Path, receipt: dict) -> None:
        return CommitStateMachine._validate_head_published_receipt(self, receipt_path, receipt)

    def _lock_key(self, raw_key: str) -> str:
        return CommitStateMachine._lock_key(self, raw_key)

    @staticmethod
    def _validate_tenant_id(value: object, label: str) -> str:
        return CommitStateMachine._validate_tenant_id(value, label)

    def _explicit_tenant_declarations(self, operation: ContextOperation) -> list[tuple[str, str]]:
        return CommitStateMachine._explicit_tenant_declarations(self, operation)

    def _operation_matches_bound_tenant(self, operation: ContextOperation) -> bool:
        return CommitStateMachine._operation_matches_bound_tenant(self, operation)

    def _validate_and_bind_operations(self, user_id: str, operations: list[ContextOperation]) -> None:
        return CommitStateMachine._validate_and_bind_operations(self, user_id, operations)

    def _validate_recovery_artifact_tenant(self, payload: object, label: str) -> None:
        return CommitStateMachine._validate_recovery_artifact_tenant(self, payload, label)

    def _validate_redo_boundary(
        self,
        user_id: str,
        operation: ContextOperation,
        *,
        source_effect: dict | None = None,
        relation_manifest: dict | None = None,
    ) -> None:
        return CommitStateMachine._validate_redo_boundary(
            self, user_id, operation, source_effect=source_effect, relation_manifest=relation_manifest
        )

    def _load_exact_redo_entry(
        self,
        user_id: str,
        operation: ContextOperation,
        phase: str,
        *,
        source_effect: dict | None,
        relation_manifest: dict | None,
    ) -> RedoEntry:
        return CommitStateMachine._load_exact_redo_entry(
            self, user_id, operation, phase, source_effect=source_effect, relation_manifest=relation_manifest
        )

    def _validate_canonical_artifact_keys(self, operation: ContextOperation) -> tuple[str, str]:
        return CommitStateMachine._validate_canonical_artifact_keys(self, operation)

    def _reject_cross_boundary_redo_collisions(self, user_id: str, operations: list[ContextOperation]) -> None:
        return CommitStateMachine._reject_cross_boundary_redo_collisions(self, user_id, operations)

    def _redo_request_matches_durable_effect(self, durable: ContextOperation, requested: ContextOperation) -> bool:
        return CommitStateMachine._redo_request_matches_durable_effect(self, durable, requested)

    def commit(self, user_id: str, operations: list[ContextOperation]) -> ContextDiff:
        return CommitCoordinator.commit(self, user_id, operations)

    def commit_ordinary_relation_update(
        self,
        *,
        owner_user_id: str,
        desired_authority: ContextObject,
        content: str,
        tenant_id: str,
    ) -> ContextDiff:
        """Commit a ContextDB relation update without exposing operation models upstream."""

        return execute_ordinary_relation_update(
            self,
            owner_user_id=owner_user_id,
            desired_authority=desired_authority,
            content=content,
            tenant_id=tenant_id,
        )

    def _commit_unfenced(self, user_id: str, operations: list[ContextOperation]) -> ContextDiff:
        return CommitCoordinator._commit_unfenced(self, user_id, operations)

    def _finalize_regular_diff(
        self,
        user_id: str,
        committed: list[ContextOperation],
        pending: list[ContextOperation],
        target_rejected: list[ContextOperation],
        conflict_rejected: list[ContextOperation],
        *,
        held_guards: list[LeaseGuard] | None = None,
    ) -> ContextDiff:
        return CommitAuditDiff._finalize_regular_diff(
            self, user_id, committed, pending, target_rejected, conflict_rejected, held_guards=held_guards
        )

    def _finalize_regular_diff_locked(
        self,
        user_id: str,
        committed: list[ContextOperation],
        pending: list[ContextOperation],
        target_rejected: list[ContextOperation],
        conflict_rejected: list[ContextOperation],
    ) -> ContextDiff:
        return CommitAuditDiff._finalize_regular_diff_locked(
            self, user_id, committed, pending, target_rejected, conflict_rejected
        )

    def _finalize_single_regular_operation(
        self, user_id: str, operation: ContextOperation, *, source_effect: dict | None, relation_manifest: dict | None
    ) -> ContextDiff:
        return CommitAuditDiff._finalize_single_regular_operation(
            self, user_id, operation, source_effect=source_effect, relation_manifest=relation_manifest
        )

    def _ensure_single_operation_diff(self, user_id: str, operation: ContextOperation) -> ContextDiff:
        return CommitAuditDiff._ensure_single_operation_diff(self, user_id, operation)

    def _validate_single_operation_diff(self, user_id: str, operation: ContextOperation) -> ContextDiff:
        return CommitAuditDiff._validate_single_operation_diff(self, user_id, operation)

    def _combine_diffs(self, user_id: str, diffs: list[ContextDiff]) -> ContextDiff:
        return CommitAuditDiff._combine_diffs(self, user_id, diffs)

    def combine_committed_diffs(self, user_id: str, diffs: list[ContextDiff]) -> ContextDiff:
        return CommitAuditDiff.combine_committed_diffs(self, user_id, diffs)

    def _commit_canonical_batch(self, user_id: str, operations: list[ContextOperation]) -> ContextDiff:
        return self._require_memory_commit_handlers().canonical_coordinator._commit_canonical_batch(
            self, user_id, operations
        )

    def _ensure_canonical_planning_digest(self, operations: list[ContextOperation], *, publish: bool = True) -> str:
        return self._require_memory_commit_handlers().canonical_planning._ensure_canonical_planning_digest(
            self, operations, publish=publish
        )

    def _ensure_pending_planning_digest(self, operation: ContextOperation) -> str:
        return self._require_memory_commit_handlers().canonical_planning._ensure_pending_planning_digest(
            self, operation
        )

    def _preflight_canonical_groups(self, user_id: str, groups: list[list[ContextOperation]]) -> None:
        return self._require_memory_commit_handlers().canonical_coordinator._preflight_canonical_groups(
            self, user_id, groups
        )

    def _validate_canonical_envelope(self, user_id: str, operations: list[ContextOperation]) -> None:
        return self._require_memory_commit_handlers().canonical_coordinator._validate_canonical_envelope(
            self, user_id, operations
        )

    def _validate_existing_canonical_boundary(self, desired: ContextObject) -> None:
        return self._require_memory_commit_handlers().canonical_coordinator._validate_existing_canonical_boundary(
            self, desired
        )

    def _reject_canonical_regular_bypass(self, operations: list[ContextOperation]) -> None:
        handlers = self._memory_commit_handlers
        if handlers is None:
            return None
        return handlers.canonical_coordinator._reject_canonical_regular_bypass(self, operations)

    def _preflight_regular_operations(
        self,
        operations: list[ContextOperation],
        *,
        validate_resolution_links: bool = True,
        validate_target_state: bool = True,
    ) -> None:
        return CommitCoordinator._preflight_regular_operations(
            self,
            operations,
            validate_resolution_links=validate_resolution_links,
            validate_target_state=validate_target_state,
        )

    def _validate_regular_operation_effect(
        self, operation: ContextOperation, *, validate_target_state: bool, allow_existing_add: bool = False
    ) -> None:
        return RegularOperationValidator._validate_regular_operation_effect(
            self, operation, validate_target_state=validate_target_state, allow_existing_add=allow_existing_add
        )

    def _validate_action_policy_operation(self, operation: ContextOperation) -> None:
        if operation.context_type != ContextType.ACTION_POLICY:
            return None
        return self._require_action_policy_commit_handlers().handler._validate_action_policy_operation(self, operation)

    def _validate_regular_canonical_boundary(
        self,
        operation: ContextOperation,
        current: ContextObject | None,
        desired: ContextObject | None,
        *,
        allow_existing_add: bool,
    ) -> None:
        handlers = self._memory_commit_handlers
        if handlers is None:
            return None
        return handlers.canonical_handler._validate_regular_canonical_boundary(
            self, operation, current, desired, allow_existing_add=allow_existing_add
        )

    def _trusted_inflight_regular_object_effect(self, operation: ContextOperation) -> ContextOperation | None:
        return RegularOperationValidator._trusted_inflight_regular_object_effect(self, operation)

    def _validate_pending_lifecycle_cas(
        self, operation: ContextOperation, *, validate_resolution_links: bool = True
    ) -> None:
        return self._require_memory_commit_handlers().canonical_planning._validate_pending_lifecycle_cas(
            self, operation, validate_resolution_links=validate_resolution_links
        )

    def _validate_pending_review_command(
        self, operation: ContextOperation, current: PendingMemoryProposal, review_binding: dict
    ) -> None:
        return self._require_memory_commit_handlers().canonical_planning._validate_pending_review_command(
            self, operation, current, review_binding
        )

    def _validate_pending_resolution_commit(self, operation: ContextOperation, pending: PendingMemoryProposal) -> None:
        return self._require_memory_commit_handlers().canonical_planning._validate_pending_resolution_commit(
            self, operation, pending
        )

    def _validate_pending_resolution_batch(self, operations: list[ContextOperation]) -> None:
        return self._require_memory_commit_handlers().canonical_planning._validate_pending_resolution_batch(
            self, operations
        )

    def _validate_pending_correction_batch(self, operations: list[ContextOperation]) -> None:
        return self._require_memory_commit_handlers().canonical_planning._validate_pending_correction_batch(
            self, operations
        )

    def _finalize_canonical_outbox(
        self,
        transaction_id: str,
        idempotency_key: str,
        operations: list[ContextOperation],
        *,
        slot_uri: str | None = None,
    ) -> Path:
        return CommitOutbox._finalize_canonical_outbox(
            self, transaction_id, idempotency_key, operations, slot_uri=slot_uri
        )

    def _preflight_canonical_revisions(
        self, operations: list[ContextOperation], *, check_revisions: bool = True
    ) -> None:
        return self._require_memory_commit_handlers().canonical_handler._preflight_canonical_revisions(
            self, operations, check_revisions=check_revisions
        )

    def _validate_canonical_evidence(self, operation: ContextOperation) -> None:
        return self._require_memory_commit_handlers().canonical_handler._validate_canonical_evidence(self, operation)

    def _same_evidence_time(self, left: object, right: object) -> bool:
        return self._require_memory_commit_handlers().canonical_handler._same_evidence_time(self, left, right)

    def _validate_authoritative_batch(self, operations: list[ContextOperation]) -> None:
        return self._require_memory_commit_handlers().canonical_handler._validate_authoritative_batch(self, operations)

    def _validate_existing_slot_invariant(self, slot_uri: str) -> None:
        return self._require_memory_commit_handlers().canonical_handler._validate_existing_slot_invariant(
            self, slot_uri
        )

    def _apply_canonical_source(self, operation: ContextOperation) -> None:
        return self._require_memory_commit_handlers().canonical_handler._apply_canonical_source(self, operation)

    def _build_canonical_relation_manifest(
        self, operation: ContextOperation, before_object: ContextObject | None
    ) -> dict:
        return CanonicalEffectExecutor._build_canonical_relation_manifest(self, operation, before_object)

    def _validate_canonical_relation_manifest(self, operation: ContextOperation, manifest: dict) -> None:
        return CanonicalEffectExecutor._validate_canonical_relation_manifest(self, operation, manifest)

    def _apply_canonical_relation_manifest(self, operation: ContextOperation, manifest: dict) -> None:
        return CanonicalEffectExecutor._apply_canonical_relation_manifest(self, operation, manifest)

    def _validate_canonical_relation_manifest_effect(self, manifest: dict) -> None:
        return CanonicalEffectExecutor._validate_canonical_relation_manifest_effect(self, manifest)

    def _canonical_relation_specs(self, operation: ContextOperation, obj: ContextObject) -> list[dict]:
        return CanonicalEffectExecutor._canonical_relation_specs(self, operation, obj)

    def _canonical_managed_relation_keys(self, obj: ContextObject | None) -> list[dict]:
        return CanonicalEffectExecutor._canonical_managed_relation_keys(self, obj)

    def _validate_existing_canonical_effect(self, operation: ContextOperation) -> None:
        return CanonicalEffectExecutor._validate_existing_canonical_effect(self, operation)

    def _capture_canonical_source_effect(self, operation: ContextOperation, relation_manifest: dict) -> dict:
        return CanonicalEffectExecutor._capture_canonical_source_effect(self, operation, relation_manifest)

    def _validate_canonical_source_effect(
        self, operation: ContextOperation, source_effect: dict | None, relation_manifest: dict | None
    ) -> None:
        return CanonicalEffectExecutor._validate_canonical_source_effect(
            self, operation, source_effect, relation_manifest
        )

    def _write_outbox_event(
        self,
        transaction_id: str,
        idempotency_key: str,
        operations: list[ContextOperation],
        *,
        status: str = "committed",
        before_images: list[dict] | None = None,
        relation_manifests: dict[str, dict] | None = None,
        receipt_path: str = "",
        receipt_digest: str = "",
    ) -> Path:
        return CommitOutbox._write_outbox_event(
            self,
            transaction_id,
            idempotency_key,
            operations,
            status=status,
            before_images=before_images,
            relation_manifests=relation_manifests,
            receipt_path=receipt_path,
            receipt_digest=receipt_digest,
        )

    def _before_image_payload(self, snapshot: dict) -> dict:
        return CommitOutbox._before_image_payload(self, snapshot)

    def _capture_canonical_state(self, operations: list[ContextOperation]) -> list[dict]:
        return CommitOutbox._capture_canonical_state(self, operations)

    def _restore_canonical_state(self, snapshots: list[dict]) -> None:
        return CommitOutbox._restore_canonical_state(self, snapshots)

    def _enqueue_outbox(
        self, transaction_id: str, slot_uri: str, outbox_path: Path, operations: list[ContextOperation]
    ) -> None:
        return CommitOutbox._enqueue_outbox(self, transaction_id, slot_uri, outbox_path, operations)

    def _transaction_marker(self, idempotency_key: str) -> Path:
        return TransactionMarkerStore._transaction_marker(self, idempotency_key)

    def _outbox_path(self, transaction_id: str) -> Path:
        return TransactionMarkerStore._outbox_path(self, transaction_id)

    def _ensure_canonical_transaction_diff(
        self, user_id: str, transaction_id: str, operations: list[ContextOperation]
    ) -> ContextDiff:
        return CommitAuditDiff._ensure_canonical_transaction_diff(self, user_id, transaction_id, operations)

    @staticmethod
    def _reject_control_symlink(path: Path, label: str) -> None:
        return TransactionMarkerStore._reject_control_symlink(path, label)

    def _write_transaction_marker(
        self,
        path: Path,
        diff: ContextDiff,
        operations: list[ContextOperation],
        *,
        relation_manifests: dict[str, dict] | None = None,
    ) -> None:
        return TransactionMarkerStore._write_transaction_marker(
            self, path, diff, operations, relation_manifests=relation_manifests
        )

    def _validate_transaction_marker(self, path: Path, operations: list[ContextOperation]) -> ContextDiff:
        return TransactionMarkerStore._validate_transaction_marker(self, path, operations)

    def _validate_transaction_marker_tenant(self, path: Path) -> None:
        return TransactionMarkerStore._validate_transaction_marker_tenant(self, path)

    def _transaction_marker_diff(self, path: Path) -> ContextDiff:
        return TransactionMarkerStore._transaction_marker_diff(self, path)

    def _marker_relation_effects(self, relation_manifests: dict[str, dict] | None) -> list[dict]:
        return TransactionMarkerStore._marker_relation_effects(self, relation_manifests)

    def _canonical_transaction_request_fingerprint(self, operations: list[ContextOperation]) -> str:
        return TransactionMarkerStore._canonical_transaction_request_fingerprint(self, operations)

    def _canonical_transaction_request_fingerprint_v2(self, operations: list[ContextOperation]) -> str:
        return TransactionMarkerStore._canonical_transaction_request_fingerprint_v2(self, operations)

    def _canonical_transaction_effect_fingerprint(self, operations: list[ContextOperation]) -> str:
        return TransactionMarkerStore._canonical_transaction_effect_fingerprint(self, operations)

    def _canonical_transaction_effect_fingerprint_v2(self, operations: list[ContextOperation]) -> str:
        return TransactionMarkerStore._canonical_transaction_effect_fingerprint_v2(self, operations)

    def _strip_relation_timestamps(self, operation_payload: dict) -> None:
        return TransactionMarkerStore._strip_relation_timestamps(self, operation_payload)

    def _context_object_without_relation_timestamps(self, value: object) -> object:
        return TransactionMarkerStore._context_object_without_relation_timestamps(self, value)

    def _diff_from_payload(self, payload: dict) -> ContextDiff:
        return CommitAuditDiff._diff_from_payload(self, payload)

    def resume(
        self,
        user_id: str,
        operation: ContextOperation,
        phase: str,
        *,
        source_effect: dict | None = None,
        relation_manifest: dict | None = None,
    ) -> bool:
        return CommitRecoveryStateMachine.resume(
            self, user_id, operation, phase, source_effect=source_effect, relation_manifest=relation_manifest
        )

    def _resume_unfenced(
        self,
        user_id: str,
        operation: ContextOperation,
        phase: str,
        *,
        source_effect: dict | None = None,
        relation_manifest: dict | None = None,
    ) -> bool:
        return CommitRecoveryStateMachine._resume_unfenced(
            self, user_id, operation, phase, source_effect=source_effect, relation_manifest=relation_manifest
        )

    def _resume_started_source_effect(
        self, user_id: str, operation: ContextOperation, *, relation_manifest: dict | None
    ) -> bool:
        return CommitRecoveryStateMachine._resume_started_source_effect(
            self, user_id, operation, relation_manifest=relation_manifest
        )

    def _resume_under_guard(
        self,
        user_id: str,
        operation: ContextOperation,
        phase: str,
        *,
        source_effect: dict | None,
        relation_manifest: dict | None,
        guard: LeaseGuard,
    ) -> bool:
        return CommitRecoveryStateMachine._resume_under_guard(
            self,
            user_id,
            operation,
            phase,
            source_effect=source_effect,
            relation_manifest=relation_manifest,
            guard=guard,
        )

    def _capture_regular_source_effect(
        self, operation: ContextOperation, relation_manifest: dict | None = None
    ) -> dict:
        return RegularEffectExecutor._capture_regular_source_effect(self, operation, relation_manifest)

    def _validate_regular_recovery_effect(
        self,
        user_id: str,
        operation: ContextOperation,
        source_effect: dict | None,
        *,
        require_relation_presence: bool = True,
        relation_manifest: dict | None = None,
    ) -> None:
        return RegularEffectExecutor._validate_regular_recovery_effect(
            self,
            user_id,
            operation,
            source_effect,
            require_relation_presence=require_relation_presence,
            relation_manifest=relation_manifest,
        )

    def _validate_and_restore_regular_recovery_effect(
        self, user_id: str, operation: ContextOperation, source_effect: dict | None, relation_manifest: dict | None
    ) -> None:
        return RegularEffectExecutor._validate_and_restore_regular_recovery_effect(
            self, user_id, operation, source_effect, relation_manifest
        )

    def _validate_regular_action_postcondition(self, operation: ContextOperation, effect: dict) -> None:
        return RegularEffectExecutor._validate_regular_action_postcondition(self, operation, effect)

    def _build_regular_relation_manifest(self, operation: ContextOperation) -> dict:
        return RegularEffectExecutor._build_regular_relation_manifest(self, operation)

    def _action_policy_source_only_relation(
        self, desired: ContextObject | None, spec: dict, eligibility: OrdinaryRelationEligibility
    ) -> bool:
        return RegularEffectExecutor._action_policy_source_only_relation(self, desired, spec, eligibility)

    def _validate_regular_relation_manifest(self, operation: ContextOperation, manifest: dict | None) -> None:
        return RegularEffectExecutor._validate_regular_relation_manifest(self, operation, manifest)

    def _apply_regular_relation_manifest(self, operation: ContextOperation, manifest: dict) -> None:
        return RegularEffectExecutor._apply_regular_relation_manifest(self, operation, manifest)

    def _validate_regular_relation_manifest_effect(self, manifest: dict) -> None:
        return RegularEffectExecutor._validate_regular_relation_manifest_effect(self, manifest)

    def _relation_spec(
        self, source_uri: str, relation_type: str, target_uri: str, metadata: dict, *, weight: float = 1.0
    ) -> dict:
        return RegularEffectExecutor._relation_spec(
            self, source_uri, relation_type, target_uri, metadata, weight=weight
        )

    def _regular_relation_has_canonical_endpoint(self, spec: dict) -> bool:
        return RegularEffectExecutor._regular_relation_has_canonical_endpoint(self, spec)

    def _ordinary_relation_eligibility(
        self, spec: dict, *, authority_uri: str = "", authority_object: ContextObject | None = None
    ) -> OrdinaryRelationEligibility:
        return RegularEffectExecutor._ordinary_relation_eligibility(
            self, spec, authority_uri=authority_uri, authority_object=authority_object
        )

    def _relation_spec_key(self, spec: dict) -> tuple[str, str, str]:
        return RegularEffectExecutor._relation_spec_key(self, spec)

    def _relation_key_payload(self, spec: dict) -> dict:
        return RegularEffectExecutor._relation_key_payload(self, spec)

    def _unique_relation_specs(self, specs: list[dict]) -> list[dict]:
        return RegularEffectExecutor._unique_relation_specs(self, specs)

    def _unique_relation_keys(self, keys: list[dict]) -> list[dict]:
        return RegularEffectExecutor._unique_relation_keys(self, keys)

    def _expected_regular_relation_specs(self, operation: ContextOperation) -> list[dict]:
        return RegularEffectExecutor._expected_regular_relation_specs(self, operation)

    def _restore_regular_relation_effect(self, operation: ContextOperation, source_effect: dict) -> None:
        return RegularEffectExecutor._restore_regular_relation_effect(self, operation, source_effect)

    def _validate_regular_relation_postcondition(self, expected: list[dict]) -> None:
        return RegularEffectExecutor._validate_regular_relation_postcondition(self, expected)

    def _regular_source_effect_uris(self, operation: ContextOperation) -> list[str]:
        return RegularEffectExecutor._regular_source_effect_uris(self, operation)

    def _regular_operation_tenant(self, operation: ContextOperation) -> str:
        return RegularEffectExecutor._regular_operation_tenant(self, operation)

    def resume_canonical_batch(self, user_id: str, entries: list) -> list[str]:
        return CommitRecoveryStateMachine.resume_canonical_batch(self, user_id, entries)

    def _resume_canonical_batch_unfenced(self, user_id: str, entries: list) -> list[str]:
        return CommitRecoveryStateMachine._resume_canonical_batch_unfenced(self, user_id, entries)

    def recover_pending_canonical(self, user_id: str, *, commit_group_id: str | None = None) -> list[str]:
        return CommitRecoveryStateMachine.recover_pending_canonical(self, user_id, commit_group_id=commit_group_id)

    def _recover_pending_canonical_unfenced(self, user_id: str, *, commit_group_id: str | None = None) -> list[str]:
        return CommitRecoveryStateMachine._recover_pending_canonical_unfenced(
            self, user_id, commit_group_id=commit_group_id
        )

    def recover_pending_regular_memory(self, user_id: str, *, commit_group_id: str) -> list[str]:
        return CommitRecoveryStateMachine.recover_pending_regular_memory(self, user_id, commit_group_id=commit_group_id)

    def _recover_pending_regular_memory_unfenced(self, user_id: str, *, commit_group_id: str) -> list[str]:
        return CommitRecoveryStateMachine._recover_pending_regular_memory_unfenced(
            self, user_id, commit_group_id=commit_group_id
        )

    def committed_canonical_diffs(self, user_id: str, commit_group_id: str) -> list[ContextDiff]:
        return CommitAuditDiff.committed_canonical_diffs(self, user_id, commit_group_id)

    def committed_memory_effect_diffs(self, user_id: str, commit_group_id: str) -> list[ContextDiff]:
        return CommitAuditDiff.committed_memory_effect_diffs(self, user_id, commit_group_id)

    def _write_recovery_diff(self, user_id: str, operation: ContextOperation) -> None:
        return CommitAuditDiff._write_recovery_diff(self, user_id, operation)

    def _operation_marker(self, operation_id: str) -> Path:
        return OperationMarkerStore._operation_marker(self, operation_id)

    @staticmethod
    def _regular_lock_keys(operation: ContextOperation) -> tuple[str, ...]:
        return OperationMarkerStore._regular_lock_keys(operation)

    def _write_operation_marker(
        self,
        operation: ContextOperation,
        *,
        source_effect: dict | None,
        relation_manifest: dict | None,
        diff: ContextDiff,
    ) -> None:
        return OperationMarkerStore._write_operation_marker(
            self, operation, source_effect=source_effect, relation_manifest=relation_manifest, diff=diff
        )

    def _bind_pending_receipt_identity(self, operation: ContextOperation) -> None:
        return OperationMarkerStore._bind_pending_receipt_identity(self, operation)

    def _publish_pending_current_head(self, path: Path, operation: ContextOperation) -> None:
        return OperationMarkerStore._publish_pending_current_head(self, path, operation)

    def _validate_operation_marker(self, path: Path, operation: ContextOperation) -> ContextOperation:
        return OperationMarkerStore._validate_operation_marker(self, path, operation)

    def _refresh_regular_effect_proofs(self, changed_uris: list[str]) -> None:
        return OperationMarkerStore._refresh_regular_effect_proofs(self, changed_uris)

    def _operation_effect_fingerprint(self, operation: ContextOperation) -> str:
        return OperationMarkerStore._operation_effect_fingerprint(self, operation)

    def _normalized_regular_object_effect(self, operation: ContextOperation) -> object:
        return OperationMarkerStore._normalized_regular_object_effect(self, operation)

    def _coalesce_non_policy_operations(self, operations: list[ContextOperation]) -> list[ContextOperation]:
        return StoreEffectWriter._coalesce_non_policy_operations(self, operations)

    def _apply_source(self, operation: ContextOperation) -> None:
        return StoreEffectWriter._apply_source(self, operation)

    def _apply_index(self, operation: ContextOperation) -> None:
        return StoreEffectWriter._apply_index(self, operation)

    def _apply_action_policy_mutation(self, policy: ActionPolicy, operation: ContextOperation) -> ActionPolicy:
        return self._require_action_policy_commit_handlers().handler._apply_action_policy_mutation(
            self, policy, operation
        )

    def _apply_supersede_source(self, operation: ContextOperation) -> None:
        return StoreEffectWriter._apply_supersede_source(self, operation)

    def _apply_supersede_index(self, operation: ContextOperation) -> None:
        return StoreEffectWriter._apply_supersede_index(self, operation)

    def _add_supersede_relations(self, old_obj: ContextObject, new_obj: ContextObject) -> None:
        return StoreEffectWriter._add_supersede_relations(self, old_obj, new_obj)

    def _read_action_policy(self, uri: str) -> ActionPolicy:
        return self._require_action_policy_commit_handlers().handler._read_action_policy(self, uri)

    def _write_action_policy(self, policy: ActionPolicy) -> None:
        return self._require_action_policy_commit_handlers().handler._write_action_policy(self, policy)

    def _materialize_action_policy_source_relations(self, obj: ContextObject) -> ContextObject:
        if obj.context_type != ContextType.ACTION_POLICY:
            return obj
        return self._require_action_policy_commit_handlers().handler._materialize_action_policy_source_relations(
            self, obj
        )

    def _apply_relations(self, obj: ContextObject, operation: ContextOperation) -> None:
        return StoreEffectWriter._apply_relations(self, obj, operation)

    def _relation_specs_for_object(self, obj: ContextObject) -> list[dict]:
        return StoreEffectWriter._relation_specs_for_object(self, obj)

    def _add_relation(self, source_uri: str, relation_type: str, target_uri: str, metadata: dict) -> None:
        return StoreEffectWriter._add_relation(self, source_uri, relation_type, target_uri, metadata)

    def _ensure_relation_specs(self, specs: list[dict]) -> None:
        return StoreEffectWriter._ensure_relation_specs(self, specs)

    def _relation_effect_spec(self, relation: ContextRelation) -> dict:
        return StoreEffectWriter._relation_effect_spec(self, relation)

    def _read_content_or_empty(self, uri: str) -> str:
        return StoreEffectWriter._read_content_or_empty(self, uri)
