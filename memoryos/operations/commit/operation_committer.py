"""Durable committer for ordinary Context and ActionPolicy objects.

Markdown memory documents are owned by ``MemoryDocumentCommitter`` and are
rejected before this class creates a redo artifact.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, TypeGuard

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.contextdb.store.index_store import IndexStore
from memoryos.contextdb.store.lock_store import LockStore
from memoryos.contextdb.store.queue_store import QueueStore
from memoryos.contextdb.store.relation_store import RelationStore
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.contextdb.transaction.path_lock import LeaseGuard, PathLock
from memoryos.core.durable_io import atomic_create_json
from memoryos.operations.commit.audit_diff import CommitAuditDiff
from memoryos.operations.commit.audit_writer import AuditWriter
from memoryos.operations.commit.coordinator import CommitCoordinator
from memoryos.operations.commit.diff_writer import DiffWriter
from memoryos.operations.commit.domain_protocols import ActionPolicy
from memoryos.operations.commit.domain_registry import (
    RegisteredActionPolicyCommitHandlers,
    action_policy_commit_handlers,
)
from memoryos.operations.commit.effects.regular import RegularEffectExecutor
from memoryos.operations.commit.effects.writer import StoreEffectWriter
from memoryos.operations.commit.markers.operation import OperationMarkerStore
from memoryos.operations.commit.operation_coalescer import OperationCoalescer
from memoryos.operations.commit.ordinary_relation import commit_ordinary_relation_update
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


def _is_lock_store(candidate: object) -> TypeGuard[LockStore]:
    required = ("acquire", "renew", "assert_owned", "fenced", "release")
    return all(callable(getattr(candidate, name, None)) for name in required)


class OperationCommitter:
    @staticmethod
    def _atomic_create_json(path, payload, *, artifact_root):  # noqa: ANN001, ANN202
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
        tombstone_service=None,  # noqa: ANN001
    ) -> None:
        source_tenant = getattr(source_store, "tenant_id", None)
        if source_tenant is not None:
            source_tenant = self._validate_tenant_id(source_tenant, "SourceStore tenant_id")
        bound = (
            self._validate_tenant_id(tenant_id, "OperationCommitter tenant_id")
            if tenant_id is not None
            else source_tenant or "default"
        )
        if source_tenant is not None and source_tenant != bound:
            raise ValueError("OperationCommitter tenant does not match SourceStore tenant")
        self.source_store = source_store
        self.index_store = index_store
        self.relation_store = relation_store
        self.queue_store = queue_store
        self.root = Path(root)
        self.artifact_root = self.root if bound == "default" else self.root / "tenants" / bound
        self.tenant_id = bound
        self.coalescer = OperationCoalescer()
        self.conflicts = ConflictResolver()
        self.target_resolver = target_resolver or TargetResolver(index_store, source_store=source_store)
        self.redo = RedoLog(self.artifact_root)
        self.diff_writer = DiffWriter(self.artifact_root)
        self.audit = AuditWriter(self.artifact_root)
        candidate = lock_store
        if candidate is None:
            provider = getattr(source_store, "operation_lock_store", None)
            candidate = provider() if callable(provider) else None
        if candidate is None:
            raise RuntimeError("OperationCommitter requires an injected LockStore")
        if not _is_lock_store(candidate):
            raise TypeError("OperationCommitter received an invalid LockStore")
        self.path_lock = PathLock(candidate)
        self._action_policy_commit_handlers = action_policy_commit_handlers()
        self.action_policy_updater = (
            self._action_policy_commit_handlers.updater_factory()
            if self._action_policy_commit_handlers is not None
            else None
        )
        self.test_hook = test_hook
        self.tombstone_service = tombstone_service
        self._startup_recovery_group: ContextVar[str] = ContextVar(
            f"memoryos_startup_recovery_group_{id(self)}", default=""
        )

    def _require_action_policy_commit_handlers(self) -> RegisteredActionPolicyCommitHandlers:
        handlers = self._action_policy_commit_handlers
        if handlers is None:
            raise RuntimeError("ActionPolicy commit handlers are not registered")
        return handlers

    @staticmethod
    def _is_document_owned_uri(uri: str) -> bool:
        parsed = ContextURI.parse(uri)
        return parsed.authority == "user" and parsed.segments[1:3] == ("memory", "documents")

    def _reject_document_owned_uri(self, uri: str) -> None:
        if self._is_document_owned_uri(uri):
            raise PermissionError(
                "Markdown memory documents cannot pass through OperationCommitter; "
                "use MemoryDocumentCommitter"
            )

    def _reject_document_owned_operation(self, operation: ContextOperation) -> None:
        if operation.context_type == ContextType.MEMORY:
            raise PermissionError(
                "ContextType.MEMORY is reserved for Markdown document projections; "
                "use MemoryDocumentCommitter"
            )
        if operation.target_uri:
            self._reject_document_owned_uri(operation.target_uri)
        raw = operation.payload.get("context_object")
        if not isinstance(raw, dict):
            return
        uri = raw.get("uri")
        if isinstance(uri, str) and uri:
            self._reject_document_owned_uri(uri)
        for relation in raw.get("relations", []) or []:
            if not isinstance(relation, dict):
                continue
            for key in ("source_uri", "target_uri"):
                endpoint = relation.get(key)
                if isinstance(endpoint, str) and endpoint.startswith("memoryos://"):
                    self._reject_document_owned_uri(endpoint)
        try:
            obj = ContextObject.from_dict(raw)
        except (KeyError, TypeError, ValueError):
            return
        for spec in self._relation_specs_for_object(obj):
            for key in ("source_uri", "target_uri"):
                endpoint = str(spec.get(key) or "")
                if endpoint.startswith("memoryos://"):
                    self._reject_document_owned_uri(endpoint)

    def commit(self, user_id: str, operations: list[ContextOperation]) -> ContextDiff:
        return CommitCoordinator.commit(self, user_id, operations)

    def _commit_unfenced(self, user_id: str, operations: list[ContextOperation]) -> ContextDiff:
        return CommitCoordinator._commit_unfenced(self, user_id, operations)

    def commit_ordinary_relation_update(
        self,
        *,
        owner_user_id: str,
        desired_authority: ContextObject,
        content: str,
        tenant_id: str,
    ) -> ContextDiff:
        return commit_ordinary_relation_update(
            self,
            owner_user_id=owner_user_id,
            desired_authority=desired_authority,
            content=content,
            tenant_id=tenant_id,
        )

    @staticmethod
    def _delete_tombstone_ids(operation: ContextOperation) -> tuple[str, ...]:
        return CommitStateMachine._delete_tombstone_ids(operation)

    def _require_delete_tombstone_capability(self, operations: list[ContextOperation]) -> None:
        CommitStateMachine._require_delete_tombstone_capability(self, operations)

    def _prepare_delete_tombstones(
        self, operation: ContextOperation, *, trust_durable_binding: bool = False
    ) -> tuple[str, ...]:
        return CommitStateMachine._prepare_delete_tombstones(
            self, operation, trust_durable_binding=trust_durable_binding
        )

    def _settle_delete_tombstones(self, operations: list[ContextOperation]) -> None:
        CommitStateMachine._settle_delete_tombstones(self, operations)

    @contextmanager
    def _durable_startup_recovery_scope(self, group_id: str) -> Iterator[None]:
        with CommitStateMachine._durable_startup_recovery_scope(self, group_id):
            yield

    def _require_commit_ready(self, user_id: str, operations: list[ContextOperation]) -> None:
        CommitStateMachine._require_commit_ready(self, user_id, operations)

    def _notify(self, stage: str, operation_id: str) -> None:
        CommitStateMachine._notify(self, stage, operation_id)

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
        CommitStateMachine._validate_and_bind_operations(self, user_id, operations)

    def _validate_recovery_artifact_tenant(self, payload: object, label: str) -> None:
        CommitStateMachine._validate_recovery_artifact_tenant(self, payload, label)

    def _validate_redo_boundary(
        self,
        user_id: str,
        operation: ContextOperation,
        *,
        source_effect: dict | None = None,
        relation_manifest: dict | None = None,
    ) -> None:
        CommitStateMachine._validate_redo_boundary(
            self,
            user_id,
            operation,
            source_effect=source_effect,
            relation_manifest=relation_manifest,
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
            self,
            user_id,
            operation,
            phase,
            source_effect=source_effect,
            relation_manifest=relation_manifest,
        )

    def _reject_cross_boundary_redo_collisions(
        self, user_id: str, operations: list[ContextOperation]
    ) -> None:
        CommitStateMachine._reject_cross_boundary_redo_collisions(self, user_id, operations)

    def _redo_request_matches_durable_effect(
        self, durable: ContextOperation, requested: ContextOperation
    ) -> bool:
        return CommitStateMachine._redo_request_matches_durable_effect(self, durable, requested)

    def _preflight_regular_operations(
        self,
        operations: list[ContextOperation],
        *,
        validate_resolution_links: bool = True,
        validate_target_state: bool = True,
    ) -> None:
        CommitCoordinator._preflight_regular_operations(
            self,
            operations,
            validate_resolution_links=validate_resolution_links,
            validate_target_state=validate_target_state,
        )

    def _validate_regular_operation_effect(
        self,
        operation: ContextOperation,
        *,
        validate_target_state: bool,
        allow_existing_add: bool = False,
    ) -> None:
        RegularOperationValidator._validate_regular_operation_effect(
            self,
            operation,
            validate_target_state=validate_target_state,
            allow_existing_add=allow_existing_add,
        )

    def _trusted_inflight_regular_object_effect(
        self, operation: ContextOperation
    ) -> ContextOperation | None:
        return RegularOperationValidator._trusted_inflight_regular_object_effect(self, operation)

    def _validate_action_policy_operation(self, operation: ContextOperation) -> None:
        if operation.context_type == ContextType.ACTION_POLICY:
            self._require_action_policy_commit_handlers().handler._validate_action_policy_operation(
                self, operation
            )

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
            self,
            user_id,
            committed,
            pending,
            target_rejected,
            conflict_rejected,
            held_guards=held_guards,
        )

    def _finalize_regular_diff_locked(self, *args) -> ContextDiff:  # noqa: ANN002
        return CommitAuditDiff._finalize_regular_diff_locked(self, *args)

    def _finalize_single_regular_operation(
        self,
        user_id: str,
        operation: ContextOperation,
        *,
        source_effect: dict | None,
        relation_manifest: dict | None,
    ) -> ContextDiff:
        return CommitAuditDiff._finalize_single_regular_operation(
            self,
            user_id,
            operation,
            source_effect=source_effect,
            relation_manifest=relation_manifest,
        )

    def _ensure_single_operation_diff(self, user_id: str, operation: ContextOperation) -> ContextDiff:
        return CommitAuditDiff._ensure_single_operation_diff(self, user_id, operation)

    def _validate_single_operation_diff(self, user_id: str, operation: ContextOperation) -> ContextDiff:
        return CommitAuditDiff._validate_single_operation_diff(self, user_id, operation)

    def _combine_diffs(self, user_id: str, diffs: list[ContextDiff]) -> ContextDiff:
        return CommitAuditDiff._combine_diffs(self, user_id, diffs)

    def combine_committed_diffs(self, user_id: str, diffs: list[ContextDiff]) -> ContextDiff:
        return CommitAuditDiff.combine_committed_diffs(self, user_id, diffs)

    def _diff_from_payload(self, payload: dict) -> ContextDiff:
        return CommitAuditDiff._diff_from_payload(self, payload)

    def _write_recovery_diff(self, user_id: str, operation: ContextOperation) -> None:
        CommitAuditDiff._write_recovery_diff(self, user_id, operation)

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
            self,
            user_id,
            operation,
            phase,
            source_effect=source_effect,
            relation_manifest=relation_manifest,
        )

    def _resume_unfenced(self, *args, **kwargs) -> bool:  # noqa: ANN002, ANN003
        return CommitRecoveryStateMachine._resume_unfenced(self, *args, **kwargs)

    def _resume_started_source_effect(self, *args, **kwargs) -> bool:  # noqa: ANN002, ANN003
        return CommitRecoveryStateMachine._resume_started_source_effect(self, *args, **kwargs)

    def _resume_under_guard(self, *args, **kwargs) -> bool:  # noqa: ANN002, ANN003
        return CommitRecoveryStateMachine._resume_under_guard(self, *args, **kwargs)

    def _capture_regular_source_effect(
        self, operation: ContextOperation, relation_manifest: dict | None = None
    ) -> dict:
        return RegularEffectExecutor._capture_regular_source_effect(self, operation, relation_manifest)

    def _validate_regular_recovery_effect(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        RegularEffectExecutor._validate_regular_recovery_effect(self, *args, **kwargs)

    def _validate_and_restore_regular_recovery_effect(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        RegularEffectExecutor._validate_and_restore_regular_recovery_effect(self, *args, **kwargs)

    def _validate_regular_action_postcondition(self, operation: ContextOperation, effect: dict) -> None:
        RegularEffectExecutor._validate_regular_action_postcondition(self, operation, effect)

    def _build_regular_relation_manifest(self, operation: ContextOperation) -> dict:
        return RegularEffectExecutor._build_regular_relation_manifest(self, operation)

    def _action_policy_source_only_relation(
        self, desired: ContextObject | None, spec: dict, eligibility: OrdinaryRelationEligibility
    ) -> bool:
        return RegularEffectExecutor._action_policy_source_only_relation(self, desired, spec, eligibility)

    def _validate_regular_relation_manifest(self, operation: ContextOperation, manifest: dict | None) -> None:
        RegularEffectExecutor._validate_regular_relation_manifest(self, operation, manifest)

    def _apply_regular_relation_manifest(self, operation: ContextOperation, manifest: dict) -> None:
        RegularEffectExecutor._apply_regular_relation_manifest(self, operation, manifest)

    def _validate_regular_relation_manifest_effect(self, manifest: dict) -> None:
        RegularEffectExecutor._validate_regular_relation_manifest_effect(self, manifest)

    def _relation_spec(self, *args, **kwargs) -> dict:  # noqa: ANN002, ANN003
        return RegularEffectExecutor._relation_spec(self, *args, **kwargs)

    def _ordinary_relation_eligibility(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return RegularEffectExecutor._ordinary_relation_eligibility(self, *args, **kwargs)

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
        RegularEffectExecutor._restore_regular_relation_effect(self, operation, source_effect)

    def _validate_regular_relation_postcondition(self, expected: list[dict]) -> None:
        RegularEffectExecutor._validate_regular_relation_postcondition(self, expected)

    def _regular_source_effect_uris(self, operation: ContextOperation) -> list[str]:
        return RegularEffectExecutor._regular_source_effect_uris(self, operation)

    def _regular_operation_tenant(self, operation: ContextOperation) -> str:
        return RegularEffectExecutor._regular_operation_tenant(self, operation)

    def _operation_marker(self, operation_id: str) -> Path:
        return OperationMarkerStore._operation_marker(self, operation_id)

    @staticmethod
    def _regular_lock_keys(operation: ContextOperation) -> tuple[str, ...]:
        return OperationMarkerStore._regular_lock_keys(operation)

    def _write_operation_marker(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        OperationMarkerStore._write_operation_marker(self, *args, **kwargs)

    def _validate_operation_marker(self, path: Path, operation: ContextOperation) -> ContextOperation:
        return OperationMarkerStore._validate_operation_marker(self, path, operation)

    def _refresh_regular_effect_proofs(self, changed_uris: list[str]) -> None:
        OperationMarkerStore._refresh_regular_effect_proofs(self, changed_uris)

    def _operation_effect_fingerprint(self, operation: ContextOperation) -> str:
        return OperationMarkerStore._operation_effect_fingerprint(self, operation)

    def _normalized_regular_object_effect(self, operation: ContextOperation) -> object:
        return OperationMarkerStore._normalized_regular_object_effect(self, operation)

    def _coalesce_non_policy_operations(self, operations: list[ContextOperation]) -> list[ContextOperation]:
        return StoreEffectWriter._coalesce_non_policy_operations(self, operations)

    def _apply_source(self, operation: ContextOperation) -> None:
        StoreEffectWriter._apply_source(self, operation)

    def _apply_index(self, operation: ContextOperation) -> None:
        StoreEffectWriter._apply_index(self, operation)

    def _apply_action_policy_mutation(
        self, policy: ActionPolicy, operation: ContextOperation
    ) -> ActionPolicy:
        return self._require_action_policy_commit_handlers().handler._apply_action_policy_mutation(
            self, policy, operation
        )

    def _read_action_policy(self, uri: str):  # noqa: ANN201
        return self._require_action_policy_commit_handlers().handler._read_action_policy(self, uri)

    def _write_action_policy(self, policy: ActionPolicy) -> None:
        self._require_action_policy_commit_handlers().handler._write_action_policy(self, policy)

    def _materialize_action_policy_source_relations(self, obj: ContextObject) -> ContextObject:
        if obj.context_type != ContextType.ACTION_POLICY:
            return obj
        return self._require_action_policy_commit_handlers().handler._materialize_action_policy_source_relations(
            self, obj
        )

    def _apply_supersede_source(self, operation: ContextOperation) -> None:
        StoreEffectWriter._apply_supersede_source(self, operation)

    def _apply_supersede_index(self, operation: ContextOperation) -> None:
        StoreEffectWriter._apply_supersede_index(self, operation)

    def _add_supersede_relations(self, old_obj: ContextObject, new_obj: ContextObject) -> None:
        StoreEffectWriter._add_supersede_relations(self, old_obj, new_obj)

    def _apply_relations(self, obj: ContextObject, operation: ContextOperation) -> None:
        StoreEffectWriter._apply_relations(self, obj, operation)

    def _relation_specs_for_object(self, obj: ContextObject) -> list[dict]:
        return StoreEffectWriter._relation_specs_for_object(self, obj)

    def _add_relation(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        StoreEffectWriter._add_relation(self, *args, **kwargs)

    def _ensure_relation_specs(self, specs: list[dict]) -> None:
        StoreEffectWriter._ensure_relation_specs(self, specs)

    def _relation_effect_spec(self, relation: ContextRelation) -> dict:
        return StoreEffectWriter._relation_effect_spec(self, relation)

    def _read_content_or_empty(self, uri: str) -> str:
        return StoreEffectWriter._read_content_or_empty(self, uri)

    @staticmethod
    def _reject_control_symlink(path: Path, label: str) -> None:
        if path.is_symlink():
            raise ValueError(f"{label} cannot be a symbolic link")


__all__ = ["OperationCommitter"]
