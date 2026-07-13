"""操作提交里的操作提交。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
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
from memoryos.contextdb.session.planning_envelope import (
    PlanningEnvelopeIntegrityError,
    PlanningEnvelopeStore,
)
from memoryos.contextdb.session.session_archive import SessionArchiveStore
from memoryos.contextdb.store.local_stores import InMemoryLockStore
from memoryos.contextdb.store.source_store import (
    IndexStore,
    LockLostError,
    LockStore,
    QueueJob,
    QueueStore,
    RelationStore,
    SourceStore,
    is_canonical_memory_object,
    is_canonical_memory_uri,
)
from memoryos.contextdb.transaction.path_lock import LeaseGuard, PathLock
from memoryos.core.ids import require_safe_path_segment, stable_hash
from memoryos.core.time import utc_now
from memoryos.memory.canonical.current_head import (
    CurrentHeadIntegrityError,
    artifact_root_for,
    load_current_head,
    publish_current_head_sets,
)
from memoryos.memory.canonical.event import canonical_digest, canonical_json, resolve_content_path
from memoryos.memory.canonical.evidence import evidence_hash
from memoryos.memory.canonical.final_state import CanonicalFinalStateValidator
from memoryos.memory.canonical.identity import IDENTITY_ALGORITHM_V2, AliasRegistry, canonical_text
from memoryos.memory.canonical.proposal import (
    PENDING_PROPOSAL_TRANSITIONS,
    MemorySemanticProposal,
    PendingMemoryProposal,
)
from memoryos.memory.canonical.review_command import (
    PendingReviewCommandIntegrityError,
    PendingReviewCommandStore,
)
from memoryos.memory.canonical.scope import ScopeRef
from memoryos.memory.canonical.state import materialized_current_revision_payload
from memoryos.memory.canonical.transaction import RevisionConflictError
from memoryos.memory.canonical.visibility import (
    committed_content,
    committed_relations,
    read_committed_canonical,
    read_committed_pending,
)
from memoryos.operations.commit.audit_writer import AuditWriter
from memoryos.operations.commit.diff_writer import DiffWriter
from memoryos.operations.commit.effect_marker import (
    EffectProofError,
    atomic_create_json,
    atomic_write_json,
    build_marker,
    normalized_relation,
    object_effect_from_store,
    relation_effects_from_manifest,
    relation_identity,
    validate_marker,
)
from memoryos.operations.commit.operation_coalescer import OperationCoalescer
from memoryos.operations.commit.outbox_envelope import (
    OutboxIntegrityError,
    assert_transition,
    build_outbox,
    planned_effect_manifest,
    validate_outbox,
)
from memoryos.operations.commit.planning_proof import (
    CANONICAL_PREPARED_INTENT_SCHEMA_VERSION,
    PENDING_PREPARED_INTENT_SCHEMA_VERSION,
    ImmutablePlanningProofStore,
    PlanningProofIntegrityError,
)
from memoryos.operations.commit.quarantine import quarantine_control_file
from memoryos.operations.commit.receipt import (
    TRANSACTION_RECEIPT_SCHEMA_VERSION,
    ReceiptIntegrityError,
    build_transaction_receipt,
    load_transaction_receipt,
    validate_transaction_receipt,
)
from memoryos.operations.commit.redo_log import RedoIntegrityError, RedoLog
from memoryos.operations.model.context_diff import ContextDiff
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus
from memoryos.operations.resolver.conflict_resolver import ConflictResolver
from memoryos.operations.resolver.target_resolver import TargetResolver


class OperationCommitter:
    """负责加锁、版本校验、批量提交、故障恢复和 Outbox 落盘。"""

    @staticmethod
    def _canonical_pending_effect(operation: ContextOperation) -> bool:
        return (
            operation.payload.get("canonical_pending_resolution") is True
            or operation.payload.get("canonical_pending_correction") is True
        )

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
        self.coalescer = OperationCoalescer()
        self.conflicts = ConflictResolver()
        self.target_resolver = target_resolver or TargetResolver(index_store, source_store=source_store)
        self.redo = RedoLog(self.artifact_root)
        self.diff_writer = DiffWriter(self.artifact_root)
        self.audit = AuditWriter(self.artifact_root)
        self.path_lock = PathLock(lock_store or InMemoryLockStore())
        self.action_policy_updater = ActionPolicyUpdater()
        self.tenant_id = bound_tenant
        self.test_hook = test_hook
        self.final_state_validator = CanonicalFinalStateValidator(
            source_store,
            relation_store,
            alias_registry,
        )
        self.planning_envelopes = PlanningEnvelopeStore(self.root, tenant_id=self.tenant_id)
        self.planning_proofs = ImmutablePlanningProofStore(self.artifact_root, tenant_id=self.tenant_id)
        self._startup_recovery_group: ContextVar[str] = ContextVar(
            f"memoryos_startup_recovery_group_{id(self)}",
            default="",
        )

    @contextmanager
    def _durable_startup_recovery_scope(self, group_id: str) -> Iterator[None]:
        """Authorize commits only for one already-durable startup group.

        The runtime builder is the sole production caller.  The final
        committer still reloads and validates the group, archive, planning
        envelope and operation bindings for every commit made in this scope.
        """

        require_safe_path_segment(group_id, "startup recovery commit_group_id")
        readiness = getattr(self.source_store, "readiness", None)
        state = getattr(getattr(readiness, "state", None), "value", "")
        if state != "RECOVERING":
            raise RuntimeError("durable startup commit scope requires a RECOVERING runtime")
        token = self._startup_recovery_group.set(group_id)
        try:
            yield
        finally:
            self._startup_recovery_group.reset(token)

    def _require_commit_ready(
        self,
        user_id: str,
        operations: list[ContextOperation],
    ) -> None:
        readiness = getattr(self.source_store, "readiness", None)
        require_ready = getattr(readiness, "require_ready", None)
        if not callable(require_ready):
            return
        state = str(getattr(getattr(readiness, "state", None), "value", ""))
        if state == "READY":
            return
        group_id = self._startup_recovery_group.get()
        if state == "RECOVERING" and group_id:
            self._validate_durable_startup_commit(group_id, user_id, operations)
            return
        require_ready()

    def _validate_durable_startup_commit(
        self,
        group_id: str,
        user_id: str,
        operations: list[ContextOperation],
    ) -> None:
        """Independently bind a RECOVERING commit to durable semantic input."""

        from memoryos.contextdb.session.commit_group import CommitGroupStore

        group = CommitGroupStore(self.artifact_root).load(group_id)
        if (
            group is None
            or group.group_id != group_id
            or group.user_id != user_id
            or group.tenant_id != self.tenant_id
            or group.complete
        ):
            raise RuntimeError("startup commit is detached from its durable commit group")
        archive = SessionArchiveStore(self.root, tenant_id=self.tenant_id).read_archive(
            group.archive_uri,
            tenant_id=self.tenant_id,
            manifest_digest=group.manifest_digest or None,
        )
        if (
            archive.task_id != group.task_id
            or archive.user_id != group.user_id
            or archive.archive_digest != group.archive_digest
            or archive.manifest_digest != group.manifest_digest
        ):
            raise RuntimeError("startup commit group is detached from its immutable archive")

        memory_operations = [
            operation
            for operation in operations
            if operation.payload.get("canonical_memory") is True
            or operation.payload.get("canonical_pending_proposal") is True
        ]
        envelope: dict | None = None
        if memory_operations:
            envelope = self.planning_envelopes.load_validated_payload(group.task_id)
            if (
                envelope.get("operation_group_identity") != group.group_id
                or envelope.get("archive_uri") != group.archive_uri
                or envelope.get("archive_digest") != group.archive_digest
                or envelope.get("manifest_digest") != group.manifest_digest
                or envelope.get("user_id") != group.user_id
                or envelope.get("tenant_id") != group.tenant_id
                or envelope.get("planning_digest") != group.planning_digest
            ):
                raise RuntimeError("startup memory commit is detached from its planning envelope")

        for index, operation in enumerate(operations):
            payload = operation.payload
            if str(payload.get("commit_group_id") or "") != group.group_id:
                raise RuntimeError("startup operation crosses its durable commit group")
            if operation.user_id != group.user_id:
                raise RuntimeError("startup operation crosses its durable archive owner")
            if operation in memory_operations:
                if envelope is None or (
                    str(payload.get("planning_task_id") or "") != group.task_id
                    or str(payload.get("planning_digest") or "") != str(envelope["planning_digest"])
                ):
                    raise RuntimeError("startup memory operation is detached from durable planning")
                continue
            consumer = str(payload.get("commit_consumer") or "")
            if consumer not in group.consumers or group.consumers[consumer].status == "completed":
                raise RuntimeError("startup derived operation has no pending durable consumer")
            expected_operation_id = f"op_{stable_hash([group.group_id, consumer, index, operation.action.value, operation.target_uri], length=32)}"
            if operation.operation_id != expected_operation_id:
                raise RuntimeError("startup derived operation identity is not deterministic")

    def _notify(self, stage: str, transaction_id: str) -> None:
        if callable(self.test_hook):
            self.test_hook(stage, transaction_id)

    def _mark_current_heads_published(
        self,
        operations: list[ContextOperation],
    ) -> None:
        """Persist the post-head crash boundary before any post-head hook."""

        for operation in operations:
            self.redo.advance(operation, phase="head_published")

    def _validate_head_published_receipt(
        self,
        receipt_path: Path,
        receipt: dict,
    ) -> None:
        """A head-published redo may never be used to recreate a missing head."""

        transaction_id = str(receipt.get("transaction_id") or "<missing>")
        for snapshot in receipt.get("effect_snapshots", []) or []:
            if not isinstance(snapshot, dict) or not snapshot.get("uri"):
                raise RedoIntegrityError("head-published receipt has an invalid effect snapshot")
            uri = str(snapshot["uri"])
            try:
                head, bound_receipt, _snapshot = load_current_head(self.artifact_root, uri)
            except (FileNotFoundError, CurrentHeadIntegrityError) as exc:
                raise RedoIntegrityError(
                    "head-published redo transaction "
                    f"{transaction_id} is missing or has an invalid current head for {uri}"
                ) from exc
            if str(head.get("current_transaction_id") or "") != transaction_id:
                object_payload = dict(snapshot.get("object", {}) or {})
                metadata = dict(object_payload.get("metadata", {}) or {})
                prior_revision = metadata.get(
                    "lifecycle_revision",
                    metadata.get("revision", 0),
                )
                current_revision = head.get("current_revision", 0)
                if (
                    isinstance(prior_revision, bool)
                    or not isinstance(prior_revision, int)
                    or isinstance(current_revision, bool)
                    or not isinstance(current_revision, int)
                    or current_revision <= prior_revision
                ):
                    raise RedoIntegrityError(
                        "head-published redo transaction "
                        f"{transaction_id} current head for {uri} is not a legal later revision"
                    )
            elif (
                str(head.get("receipt_digest") or "") != str(receipt.get("receipt_digest") or "")
                or str(bound_receipt.get("receipt_digest") or "") != str(receipt.get("receipt_digest") or "")
            ):
                raise RedoIntegrityError(
                    "head-published redo transaction "
                    f"{transaction_id} current head for {uri} is detached from its receipt"
                )
            try:
                # A current head is a proof of the complete live bundle, not
                # merely a pointer to an immutable receipt.  This committed
                # read also preserves an older snapshot when a separately
                # proved pre-head transaction is legitimately in flight.
                read_committed_canonical(self.source_store, uri, self.relation_store)
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                raise RedoIntegrityError(
                    "head-published redo transaction "
                    f"{transaction_id} current Source bundle for {uri} is invalid"
                ) from exc
        if not receipt_path.exists():
            raise RedoIntegrityError(
                f"head-published redo transaction {transaction_id} is missing its immutable receipt"
            )

    def _lock_key(self, raw_key: str) -> str:
        # The default tenant keeps its historical lock key. Non-default
        # tenants have physically distinct artifacts and therefore receive a
        # tenant-qualified key in the shared lock store.
        canonical_key = raw_key
        if raw_key.startswith("memoryos://"):
            canonical_key = str(ContextURI.parse(raw_key))
        return canonical_key if self.tenant_id == "default" else f"tenant:{self.tenant_id}:{canonical_key}"

    @staticmethod
    def _validate_tenant_id(value: object, label: str) -> str:
        if not isinstance(value, str) or not value.strip() or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"{label} must be one safe non-empty path segment")
        return value

    def _explicit_tenant_declarations(self, operation: ContextOperation) -> list[tuple[str, str]]:
        payload = operation.payload
        if not isinstance(payload, dict):
            raise ValueError("operation payload must be an object")
        declarations: list[tuple[str, str]] = []

        def inspect(container: object, path: str) -> None:
            if not isinstance(container, dict) or "tenant_id" not in container:
                return
            declarations.append((path, self._validate_tenant_id(container["tenant_id"], f"{path}.tenant_id")))

        def inspect_scope(container: object, path: str) -> None:
            if not isinstance(container, dict):
                return
            inspect(container, path)
            inspect(container.get("visibility"), f"{path}.visibility")
            inspect(container.get("authority"), f"{path}.authority")

        inspect(payload, "payload")
        inspect_scope(payload.get("scope"), "payload.scope")
        inspect(payload.get("visibility"), "payload.visibility")
        inspect(payload.get("authority"), "payload.authority")
        object_payload = payload.get("context_object")
        if isinstance(object_payload, dict):
            inspect(object_payload, "payload.context_object")
            inspect_scope(object_payload.get("scope"), "payload.context_object.scope")
            inspect(object_payload.get("visibility"), "payload.context_object.visibility")
            inspect(object_payload.get("authority"), "payload.context_object.authority")
            metadata = object_payload.get("metadata")
            if isinstance(metadata, dict):
                inspect(metadata, "payload.context_object.metadata")
                inspect_scope(metadata.get("scope"), "payload.context_object.metadata.scope")
                inspect(metadata.get("visibility"), "payload.context_object.metadata.visibility")
                inspect(metadata.get("authority"), "payload.context_object.metadata.authority")
        return declarations

    def _operation_matches_bound_tenant(self, operation: ContextOperation) -> bool:
        try:
            declarations = self._explicit_tenant_declarations(operation)
        except ValueError:
            return False
        return all(value == self.tenant_id for _, value in declarations)

    def _validate_and_bind_operations(
        self,
        user_id: str,
        operations: list[ContextOperation],
    ) -> None:
        """Validate the complete principal/tenant boundary before durable effects."""

        require_safe_path_segment(user_id, "commit user_id")
        declarations_by_operation: list[tuple[ContextOperation, list[tuple[str, str]]]] = []
        for operation in operations:
            require_safe_path_segment(operation.operation_id, "operation_id")
            if operation.user_id != user_id:
                raise ValueError("operation user does not match commit user")
            declarations = self._explicit_tenant_declarations(operation)
            if any(value != self.tenant_id for _, value in declarations):
                paths = ", ".join(path for path, value in declarations if value != self.tenant_id)
                raise ValueError(f"operation tenant does not match bound tenant: {paths}")
            declarations_by_operation.append((operation, declarations))

        # Bind only after every operation has passed so a rejected batch is not
        # partially normalized and no artifact is written with an implicit tenant.
        for operation, _ in declarations_by_operation:
            operation.payload.setdefault("tenant_id", self.tenant_id)
            object_payload = operation.payload.get("context_object")
            if isinstance(object_payload, dict):
                object_payload.setdefault("tenant_id", self.tenant_id)

    def _validate_recovery_artifact_tenant(self, payload: object, label: str) -> None:
        if not isinstance(payload, dict) or "tenant_id" not in payload:
            return
        tenant = self._validate_tenant_id(payload["tenant_id"], f"{label} tenant_id")
        if tenant != self.tenant_id:
            raise RedoIntegrityError(f"{label} crosses the bound tenant")

    def _validate_redo_boundary(
        self,
        user_id: str,
        operation: ContextOperation,
        *,
        source_effect: dict | None = None,
        relation_manifest: dict | None = None,
    ) -> None:
        try:
            self._validate_and_bind_operations(user_id, [operation])
        except ValueError as exc:
            raise RedoIntegrityError("redo operation crosses its user or tenant boundary") from exc
        try:
            # Recovery is an alternate write entry point, so it must enforce
            # the same canonical/pending classification as a fresh commit.
            # Otherwise a legacy or hand-built regular redo could rewrite a
            # receipt-backed object before regular postcondition validation
            # notices the incompatible materialization.
            self._reject_canonical_regular_bypass([operation])
        except ValueError as exc:
            raise RedoIntegrityError(
                "canonical memory redo recovery cannot bypass its committed transaction boundary"
            ) from exc
        self._validate_recovery_artifact_tenant(source_effect, "redo source effect")
        self._validate_recovery_artifact_tenant(relation_manifest, "redo relation manifest")

    def _validate_canonical_artifact_keys(self, operation: ContextOperation) -> tuple[str, str]:
        transaction_id = require_safe_path_segment(
            operation.payload.get("transaction_id"),
            "canonical transaction_id",
        )
        idempotency_key = require_safe_path_segment(
            operation.payload.get("idempotency_key"),
            "canonical idempotency_key",
        )
        return transaction_id, idempotency_key

    def _reject_cross_boundary_redo_collisions(
        self,
        user_id: str,
        operations: list[ContextOperation],
    ) -> None:
        requested_ids = {operation.operation_id for operation in operations}
        if not requested_ids:
            return
        for entry in self.redo.pending_entries():
            if entry.operation_id not in requested_ids:
                continue
            if entry.operation.user_id != user_id or not self._operation_matches_bound_tenant(entry.operation):
                raise RedoIntegrityError("redo operation id is already bound to another user or tenant")

    def commit(self, user_id: str, operations: list[ContextOperation]) -> ContextDiff:
        """执行这一步处理，并保持已有状态约束。"""

        self._require_commit_ready(user_id, operations)
        self._validate_and_bind_operations(user_id, operations)
        self._reject_control_symlink(
            self.artifact_root / "system" / "audit" / f"{user_id}.jsonl",
            "audit control file",
        )
        self._reject_cross_boundary_redo_collisions(user_id, operations)
        self._reject_canonical_regular_bypass(operations)
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
        with ExitStack() as lock_stack:
            guard_by_key = {
                lock_key: lock_stack.enter_context(self.path_lock.acquire(self._lock_key(lock_key)))
                for lock_key in sorted(
                    {
                        lock_key
                        for operation in conflict_result.accepted
                        if operation.status != OperationStatus.PENDING
                        for lock_key in self._regular_lock_keys(operation)
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
                    operation_guards = [guard_by_key[lock_key] for lock_key in self._regular_lock_keys(operation)]
                    guard = guard_by_key[target_lock_key]
                    with self.path_lock.fenced(operation_guards):
                        marker = self._operation_marker(operation.operation_id)
                        self._reject_control_symlink(marker, "operation receipt")
                        pending_entry = pending_redo.get(operation.operation_id)
                        if pending_entry is not None and pending_entry.phase not in {"started", "begin"}:
                            self._resume_under_guard(
                                user_id,
                                pending_entry.operation,
                                pending_entry.phase,
                                source_effect=pending_entry.source_effect,
                                relation_manifest=pending_entry.relation_manifest,
                                guard=guard,
                            )
                            if marker.exists():
                                persisted = self._validate_operation_marker(marker, operation)
                                if persisted.payload.get("canonical_pending_proposal") is True:
                                    self._validate_head_published_receipt(
                                        marker,
                                        load_transaction_receipt(marker),
                                    )
                                self._ensure_single_operation_diff(user_id, persisted)
                                operation.status = OperationStatus.COMMITTED
                                committed.append(persisted)
                                continue
                        if marker.exists():
                            persisted = self._validate_operation_marker(marker, operation)
                            if persisted.payload.get("canonical_pending_proposal") is True:
                                self._validate_head_published_receipt(
                                    marker,
                                    load_transaction_receipt(marker),
                                )
                            self._ensure_single_operation_diff(user_id, persisted)
                            operation.status = OperationStatus.COMMITTED
                            committed.append(persisted)
                            continue
                        self._validate_pending_lifecycle_cas(operation)
                        relation_manifest = self._build_regular_relation_manifest(operation)
                        if operation.payload.get("canonical_pending_proposal") is True:
                            try:
                                self.planning_proofs.ensure_pending_intent(
                                    operation,
                                    relation_manifest=relation_manifest,
                                )
                            except PlanningProofIntegrityError as exc:
                                raise ValueError("pending lifecycle prepared intent is invalid") from exc
                        self.redo.begin(
                            operation,
                            phase="started",
                            relation_manifest=relation_manifest,
                        )
                        self._apply_source(operation)
                        self._apply_regular_relation_manifest(operation, relation_manifest)
                        source_effect = self._capture_regular_source_effect(
                            operation,
                            relation_manifest,
                        )
                        self._validate_regular_recovery_effect(
                            user_id,
                            operation,
                            source_effect,
                            relation_manifest=relation_manifest,
                        )
                        self.redo.advance(
                            operation,
                            phase="source_written",
                            source_effect=source_effect,
                            relation_manifest=relation_manifest,
                        )
                    with self.path_lock.fenced(operation_guards):
                        self._apply_index(operation)
                        self.redo.advance(operation, phase="index_written")
                    with self.path_lock.fenced(operation_guards):
                        self.audit.record(user_id, "context_operation_committed", operation.to_dict())
                        self.redo.advance(operation, phase="audit_written")
                        operation.status = OperationStatus.COMMITTED
                        self._finalize_single_regular_operation(
                            user_id,
                            operation,
                            source_effect=source_effect,
                            relation_manifest=relation_manifest,
                        )
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
                            held_guards=held_guards,
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
                held_guards=held_guards,
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
        if held_guards is not None:
            with self.path_lock.fenced(held_guards):
                return self._finalize_regular_diff_locked(
                    user_id,
                    committed,
                    pending,
                    target_rejected,
                    conflict_rejected,
                )
        guards = []
        with ExitStack() as lock_stack:
            lock_keys = sorted({lock_key for operation in committed for lock_key in self._regular_lock_keys(operation)})
            for lock_key in lock_keys:
                guards.append(lock_stack.enter_context(self.path_lock.acquire(self._lock_key(lock_key))))
            with self.path_lock.fenced(guards):
                return self._finalize_regular_diff_locked(
                    user_id,
                    committed,
                    pending,
                    target_rejected,
                    conflict_rejected,
                )

    def _finalize_regular_diff_locked(
        self,
        user_id: str,
        committed: list[ContextOperation],
        pending: list[ContextOperation],
        target_rejected: list[ContextOperation],
        conflict_rejected: list[ContextOperation],
    ) -> ContextDiff:
        for operation in committed:
            marker = self._operation_marker(operation.operation_id)
            self._reject_control_symlink(marker, "operation receipt")
            if not marker.exists():
                raise RedoIntegrityError("combined regular diff contains an unmarked Source effect")
            self._validate_operation_marker(marker, operation)
            self._validate_single_operation_diff(user_id, operation)
        diff_members = [*committed, *pending, *target_rejected, *conflict_rejected]
        diff_key = stable_hash(
            sorted(
                (
                    operation.operation_id,
                    operation.status.value,
                    self._operation_effect_fingerprint(operation),
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
        diff_path = self.artifact_root / "system" / "diffs" / f"{diff_id}.json"
        self._reject_control_symlink(diff_path, "regular diff artifact")
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
            for kind in ("operations", "pending_operations", "rejected_operations"):
                persisted_by_id = {item.operation_id: item for item in getattr(persisted, kind)}
                if any(
                    self._operation_effect_fingerprint(operation)
                    != self._operation_effect_fingerprint(persisted_by_id[operation.operation_id])
                    for operation in getattr(diff, kind)
                ):
                    raise ValueError("regular diff conflicts with a different persisted effect")
            diff = persisted
        else:
            self.diff_writer.write(diff)
        return diff

    def _finalize_single_regular_operation(
        self,
        user_id: str,
        operation: ContextOperation,
        *,
        source_effect: dict | None,
        relation_manifest: dict | None,
    ) -> ContextDiff:
        if operation.payload.get("canonical_pending_proposal") is True:
            self._bind_pending_receipt_identity(operation)
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
        self,
        user_id: str,
        operation: ContextOperation,
    ) -> ContextDiff:
        operation_id = require_safe_path_segment(operation.operation_id, "operation_id")
        path = self.artifact_root / "system" / "diffs" / f"diff_{operation_id}.json"
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
        self.diff_writer.write(diff)
        return diff

    def _validate_single_operation_diff(
        self,
        user_id: str,
        operation: ContextOperation,
    ) -> ContextDiff:
        operation_id = require_safe_path_segment(operation.operation_id, "operation_id")
        path = self.artifact_root / "system" / "diffs" / f"diff_{operation_id}.json"
        self._reject_control_symlink(path, "single-operation diff artifact")
        if not path.exists():
            raise RedoIntegrityError("committed regular operation has no single-operation diff")
        diff = self._diff_from_payload(json.loads(path.read_text(encoding="utf-8")))
        if (
            diff.user_id != user_id
            or len(diff.operations) != 1
            or diff.pending_operations
            or diff.rejected_operations
            or diff.operations[0].operation_id != operation.operation_id
            or self._operation_effect_fingerprint(diff.operations[0]) != self._operation_effect_fingerprint(operation)
        ):
            raise RedoIntegrityError("single-operation diff conflicts with its committed effect")
        return diff

    def _combine_diffs(self, user_id: str, diffs: list[ContextDiff]) -> ContextDiff:
        for diff in diffs:
            if diff.user_id != user_id:
                raise ValueError("committed diff crosses a user boundary")
            self._validate_and_bind_operations(
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
        combined_id = require_safe_path_segment(combined.diff_id, "diff_id")
        path = self.artifact_root / "system" / "diffs" / f"{combined_id}.json"
        self._reject_control_symlink(path, "combined diff artifact")
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
        self._reject_control_symlink(completed, "canonical transaction receipt")
        pending_entries = [
            entry
            for entry in self.redo.pending_entries()
            if str(entry.operation.payload.get("transaction_id") or "") == transaction_id
        ]
        if completed.exists() and pending_entries:
            self.resume_canonical_batch(user_id, pending_entries)
            return self._validate_transaction_marker(completed, operations)

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
                if self._canonical_pending_effect(operation) and operation.target_uri
            ),
        }
        with ExitStack() as locks:
            guards: list[LeaseGuard] = []
            for lock_key in sorted(lock_keys):
                guards.append(locks.enter_context(self.path_lock.acquire(self._lock_key(lock_key))))
            with self.path_lock.fenced(guards):
                if completed.exists():
                    diff = self._validate_transaction_marker(completed, operations)
                    self._ensure_canonical_planning_digest(operations)
                    receipt = load_transaction_receipt(completed)
                    self._validate_head_published_receipt(completed, receipt)
                    self._finalize_canonical_outbox(transaction_id, idempotency_key, diff.operations)
                    return diff
                self._preflight_canonical_revisions(operations)
                self._validate_authoritative_batch(operations)
                self.final_state_validator.validate(
                    operations,
                    tenant_id=self.tenant_id,
                    owner_user_id=user_id,
                )
                self._ensure_canonical_planning_digest(operations)
                backups = self._capture_canonical_state(operations)
                before_by_uri = {
                    str(snapshot["uri"]): (
                        snapshot["object"] if isinstance(snapshot.get("object"), ContextObject) else None
                    )
                    for snapshot in backups
                }
                relation_manifests = {
                    operation.operation_id: self._build_canonical_relation_manifest(
                        operation,
                        before_by_uri.get(str(operation.target_uri or "")),
                    )
                    for operation in operations
                }
                committed: list[ContextOperation] = []
                self._notify("before_redo", transaction_id)
                self._write_outbox_event(
                    transaction_id,
                    idempotency_key,
                    operations,
                    status="prepared",
                    before_images=backups,
                    relation_manifests=relation_manifests,
                )
                for operation in operations:
                    self.redo.begin(
                        operation,
                        phase="started",
                        relation_manifest=relation_manifests[operation.operation_id],
                    )
                self._notify("after_redo_begin", transaction_id)
            try:
                for operation in operations:
                    with self.path_lock.fenced(guards):
                        self._apply_canonical_source(operation)
                        self._notify("after_source_effect", transaction_id)
                        self._apply_canonical_relation_manifest(
                            operation,
                            relation_manifests[operation.operation_id],
                        )
                        self._notify("after_relation_effect", transaction_id)
                        source_effect = self._capture_canonical_source_effect(
                            operation,
                            relation_manifests[operation.operation_id],
                        )
                        self.redo.advance(
                            operation,
                            phase="source_written",
                            source_effect=source_effect,
                            relation_manifest=relation_manifests[operation.operation_id],
                        )
                        self.audit.record(user_id, "canonical_memory_operation_applied", operation.to_dict())
                        self._notify("after_audit", transaction_id)
                        self.redo.advance(operation, phase="audit_written")
                        operation.status = OperationStatus.COMMITTED
                        committed.append(operation)
                with self.path_lock.fenced(guards):
                    self._write_outbox_event(
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
                with self.path_lock.fenced(guards):
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
            with self.path_lock.fenced(guards):
                diff = self._ensure_canonical_transaction_diff(
                    user_id,
                    transaction_id,
                    committed,
                )
                self._notify("after_diff", transaction_id)
                self._notify("before_receipt", transaction_id)
                self._write_transaction_marker(
                    completed,
                    diff,
                    committed,
                    relation_manifests=relation_manifests,
                )
                receipt = load_transaction_receipt(completed)
                self._notify("after_receipt", transaction_id)
                self._notify("before_current_head", transaction_id)
                publish_current_head_sets(self.artifact_root, completed, receipt)
                self._mark_current_heads_published(committed)
                self._notify("after_current_head", transaction_id)
                self.audit.record(
                    user_id,
                    "canonical_memory_transaction_committed",
                    {"transaction_id": transaction_id, "operation_ids": [item.operation_id for item in committed]},
                )
                self._finalize_canonical_outbox(transaction_id, idempotency_key, committed, slot_uri=slot_uri)
                self._notify("before_redo_cleanup", transaction_id)
                for operation in committed:
                    self.redo.commit(operation.operation_id)
                return diff

    def _ensure_canonical_planning_digest(
        self,
        operations: list[ContextOperation],
        *,
        publish: bool = True,
    ) -> str:
        declared = {
            str(operation.payload.get("planning_digest") or "")
            for operation in operations
            if operation.payload.get("planning_digest")
        }
        if len(declared) > 1:
            raise ValueError("canonical transaction contains multiple planning digests")
        task_ids = {str(operation.payload.get("planning_task_id") or "") for operation in operations} - {""}
        if len(task_ids) > 1:
            raise ValueError("canonical transaction crosses planning task identities")
        task_id = next(iter(task_ids), "")
        proof_operations = [operation for operation in operations if not self._canonical_pending_effect(operation)]
        if not proof_operations:
            raise ValueError("canonical transaction has no domain operation proposal proof")
        proof_payloads: dict[str, dict] = {}
        proof_sets: set[str] = set()
        missing_proof_count = 0
        for operation in proof_operations:
            raw_proofs = operation.payload.get("proposal_proofs")
            if (
                not isinstance(raw_proofs, list)
                or not raw_proofs
                or any(not isinstance(item, dict) for item in raw_proofs)
            ):
                missing_proof_count += 1
                continue
            proof_sets.add(canonical_json(raw_proofs))
            for raw in raw_proofs:
                try:
                    proposal = MemorySemanticProposal.from_dict(raw)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("canonical transaction proposal proof is invalid") from exc
                fingerprint = proposal.fingerprint
                if fingerprint in proof_payloads and canonical_json(proof_payloads[fingerprint]) != canonical_json(raw):
                    raise ValueError("canonical transaction contains conflicting proposal proofs")
                proof_payloads[fingerprint] = raw
            declared_fingerprints = {str(item) for item in operation.payload.get("proposal_fingerprints", []) or []}
            if declared_fingerprints != set(proof_payloads):
                raise ValueError("canonical transaction proposal proof identity is inconsistent")
        if missing_proof_count not in {0, len(proof_operations)}:
            raise ValueError("canonical transaction has a partial proposal proof set")
        if not missing_proof_count and len(proof_sets) != 1:
            raise ValueError("canonical transaction operations disagree on proposal proof")
        envelope_path = self.planning_envelopes.path(task_id) if task_id else None
        anchor_path = self.planning_envelopes.anchor_path(task_id) if task_id else None
        if task_id and (
            (envelope_path is not None and (envelope_path.exists() or envelope_path.is_symlink()))
            or (anchor_path is not None and (anchor_path.exists() or anchor_path.is_symlink()))
        ):
            if missing_proof_count:
                raise ValueError("canonical transaction has no complete proposal proof")
            try:
                envelope = self.planning_envelopes.load_validated_payload(task_id)
            except PlanningEnvelopeIntegrityError as exc:
                raise ValueError("canonical transaction planning envelope is invalid") from exc
            commit_groups = {
                str(operation.payload.get("commit_group_id") or "")
                for operation in operations
                if operation.payload.get("commit_group_id")
            }
            fingerprints = {
                str(value)
                for operation in operations
                for value in operation.payload.get("proposal_fingerprints", []) or []
            }
            envelope_fingerprints = {str(value) for value in envelope.get("proposal_fingerprints", []) or []}
            envelope_proofs = {
                MemorySemanticProposal.from_dict(dict(item.get("proposal", {}) or {})).fingerprint: dict(
                    item.get("proposal", {}) or {}
                )
                for item in envelope.get("proposal_inputs", []) or []
                if isinstance(item, dict) and isinstance(item.get("proposal"), dict)
            }
            digest = str(envelope["planning_digest"])
            if (
                len(commit_groups) > 1
                or (commit_groups and commit_groups != {str(envelope.get("operation_group_identity") or "")})
                or not fingerprints.issubset(envelope_fingerprints)
                or set(proof_payloads) != fingerprints
                or any(
                    fingerprint not in envelope_proofs
                    or canonical_json(payload) != canonical_json(envelope_proofs[fingerprint])
                    for fingerprint, payload in proof_payloads.items()
                )
                or (declared and declared != {digest})
            ):
                raise ValueError("canonical transaction is detached from its durable planning envelope")
        else:
            transaction_ids = {str(operation.payload.get("transaction_id") or "") for operation in operations}
            idempotency_keys = {str(operation.payload.get("idempotency_key") or "") for operation in operations}
            commit_groups = {
                str(operation.payload.get("commit_group_id") or "")
                for operation in operations
                if operation.payload.get("commit_group_id")
            }
            if (
                len(transaction_ids) != 1
                or "" in transaction_ids
                or len(idempotency_keys) != 1
                or "" in idempotency_keys
            ):
                raise ValueError("direct canonical plan has invalid transaction identity")
            transaction_id = next(iter(transaction_ids))
            idempotency_key = next(iter(idempotency_keys))
            marker_path = self._transaction_marker(idempotency_key)
            self._reject_control_symlink(marker_path, "canonical transaction receipt")
            try:
                if marker_path.exists():
                    proof = self.planning_proofs.load_direct(
                        transaction_id,
                        operations=operations,
                    )
                elif publish:
                    proof = self.planning_proofs.ensure_direct(
                        operations,
                        kind="canonical",
                        transaction_id=transaction_id,
                        idempotency_key=idempotency_key,
                        user_id=operations[0].user_id,
                        commit_group_id=next(iter(commit_groups), ""),
                    )
                else:
                    proof = self.planning_proofs.build_direct(
                        operations,
                        kind="canonical",
                        transaction_id=transaction_id,
                        idempotency_key=idempotency_key,
                        user_id=operations[0].user_id,
                        commit_group_id=next(iter(commit_groups), ""),
                    )
            except PlanningProofIntegrityError as exc:
                raise ValueError("canonical transaction has no valid immutable planning proof") from exc
            digest = str(proof["planning_digest"])
        for operation in operations:
            operation.payload["planning_digest"] = digest
        return digest

    def _ensure_pending_planning_digest(self, operation: ContextOperation) -> str:
        task_id = str(operation.payload.get("planning_task_id") or "")
        if task_id and (
            self.planning_envelopes.path(task_id).exists()
            or self.planning_envelopes.path(task_id).is_symlink()
            or self.planning_envelopes.anchor_path(task_id).exists()
            or self.planning_envelopes.anchor_path(task_id).is_symlink()
        ):
            try:
                envelope = self.planning_envelopes.load_validated_payload(task_id)
            except PlanningEnvelopeIntegrityError as exc:
                raise ValueError("pending lifecycle planning envelope is invalid") from exc
            proposal_id = str(operation.payload.get("pending_proposal_id") or "")
            envelope_proposal_ids = {
                str(dict(item.get("proposal", {}) or {}).get("proposal_id") or "")
                for item in envelope.get("proposal_inputs", []) or []
                if isinstance(item, dict)
            }
            digest = str(envelope["planning_digest"])
            declared = str(operation.payload.get("planning_digest") or "")
            if (
                str(operation.payload.get("commit_group_id") or "")
                != str(envelope.get("operation_group_identity") or "")
                or (proposal_id and proposal_id not in envelope_proposal_ids)
                or (declared and declared != digest)
            ):
                raise ValueError("pending lifecycle is detached from its durable planning envelope")
        else:
            marker_path = self._operation_marker(operation.operation_id)
            self._reject_control_symlink(marker_path, "pending operation receipt")
            try:
                if marker_path.exists():
                    proof = self.planning_proofs.load_direct(
                        operation.operation_id,
                        operations=[operation],
                    )
                else:
                    proof = self.planning_proofs.ensure_direct(
                        [operation],
                        kind="pending",
                        transaction_id=operation.operation_id,
                        idempotency_key=str(operation.payload.get("idempotency_key") or operation.operation_id),
                        user_id=operation.user_id,
                        commit_group_id=str(operation.payload.get("commit_group_id") or ""),
                    )
            except PlanningProofIntegrityError as exc:
                raise ValueError("pending lifecycle has no valid immutable planning proof") from exc
            digest = str(proof["planning_digest"])
        operation.payload["planning_digest"] = digest
        return digest

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
            self._reject_control_symlink(
                self.artifact_root / "system" / "diffs" / f"diff_{transaction_id}.json",
                "canonical diff artifact",
            )
            existing_transaction = idempotency_transactions.setdefault(idempotency_key, transaction_id)
            if existing_transaction != transaction_id:
                raise ValueError("canonical idempotency key cannot identify multiple transactions")
            self._canonical_transaction_request_fingerprint(operations)
            self._canonical_transaction_effect_fingerprint(operations)
            marker = self._transaction_marker(idempotency_key)
            self._reject_control_symlink(marker, "canonical transaction receipt")
            if marker.exists():
                planning_error: ValueError | None = None
                try:
                    self._ensure_canonical_planning_digest(operations)
                except ValueError as exc:
                    planning_error = exc
                self._validate_transaction_marker(marker, operations)
                if planning_error is not None:
                    raise planning_error
                continue
            for operation in operations:
                if self._canonical_pending_effect(operation):
                    self._validate_pending_lifecycle_cas(operation, validate_resolution_links=False)
            self._validate_pending_resolution_batch(operations)
            self._validate_pending_correction_batch(operations)
            self._preflight_canonical_revisions(operations, check_revisions=False)
            self._validate_authoritative_batch(operations)
            for operation in operations:
                if self._canonical_pending_effect(operation):
                    continue
                object_payload = operation.payload.get("context_object")
                assert isinstance(object_payload, dict)
                uri = str(object_payload["uri"])
                if uri not in virtual_revisions:
                    try:
                        current = read_committed_canonical(
                            self.source_store,
                            uri,
                            self.relation_store,
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
            self._ensure_canonical_planning_digest(operations, publish=False)

    def _validate_canonical_envelope(self, user_id: str, operations: list[ContextOperation]) -> None:
        """Validate immutable ownership boundaries before any marker fast path."""

        self._validate_and_bind_operations(user_id, operations)
        if not user_id:
            raise ValueError("canonical commit requires a user_id")
        for operation in operations:
            self._validate_canonical_artifact_keys(operation)
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
            operation_tenant = str(operation.payload.get("tenant_id") or self.tenant_id)
            object_tenant = str(obj.tenant_id or self.tenant_id)
            if object_tenant != operation_tenant:
                raise ValueError("canonical context object tenant does not match operation tenant")
            if operation_tenant != self.tenant_id:
                raise ValueError("canonical operation tenant does not match bound tenant")
            metadata = dict(obj.metadata or {})
            if self._canonical_pending_effect(operation):
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
                self._validate_existing_canonical_boundary(obj)

    def _validate_existing_canonical_boundary(self, desired: ContextObject) -> None:
        try:
            current = read_committed_canonical(
                self.source_store,
                desired.uri,
                self.relation_store,
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

    def _reject_canonical_regular_bypass(self, operations: list[ContextOperation]) -> None:
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
                    existing = self.source_store.read_object(target)
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
            if operation.payload.get("canonical_pending_proposal") is True:
                self._bind_pending_receipt_identity(operation)
            self._reject_control_symlink(
                self.artifact_root / "system" / "diffs" / f"diff_{operation.operation_id}.json",
                "single-operation diff artifact",
            )
            marker = self._operation_marker(operation.operation_id)
            self._reject_control_symlink(marker, "operation receipt")
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
            if operation.payload.get("canonical_pending_proposal") is True:
                self._ensure_pending_planning_digest(operation)

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
            operation.payload.get("canonical_pending_proposal") is True or lifecycle_transition or desired_pending
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
        if not isinstance(expected_payload, dict) or canonical_json(current.to_dict()) != canonical_json(
            expected_payload
        ):
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
        committed_pending = read_committed_pending(
            self.source_store,
            target_uri,
            self.relation_store,
        )
        current_obj = committed_pending.object
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
        decision_for_state = {
            LifecycleState.CONFIRMED: "CONFIRM",
            LifecycleState.RESOLVED: "CONFIRM_AND_APPLY",
            LifecycleState.RETRYABLE: "RETRY",
            LifecycleState.REJECTED: "REJECT",
            LifecycleState.EXPIRED: "EXPIRE",
        }.get(desired.lifecycle_state)
        if decision_for_state is not None:
            current.assert_review_decision(decision_for_state)
        if (
            current_obj.uri != target_uri
            or current_obj.context_type != ContextType.MEMORY
            or current_obj.owner_user_id != operation.user_id
            or desired_obj.owner_user_id != current_obj.owner_user_id
            or str(desired_obj.tenant_id or "default") != str(current_obj.tenant_id or "default")
            or desired_obj.context_type != current_obj.context_type
        ):
            raise ValueError("pending lifecycle transition cannot change owner, tenant, URI, or context type")
        if (
            current_obj.lifecycle_state != current.lifecycle_state
            or desired_obj.lifecycle_state != desired.lifecycle_state
        ):
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
            current_content = (
                committed_pending.content_override
                if committed_pending.content_override is not None
                else self.source_store.read_content(current_obj.layers.l2_uri or current_obj.uri)
            )
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
        review_binding = operation.payload.get("pending_review_binding", {})
        if not isinstance(review_binding, dict):
            raise ValueError("pending lifecycle review binding must be an object")
        if decision_for_state is not None:
            self._validate_pending_review_command(
                operation,
                current,
                review_binding,
            )
        if review_binding:
            if set(review_binding) != {"command_id", "decision", "request_digest"}:
                raise ValueError("pending lifecycle review binding has unexpected fields")
            command_id = str(review_binding.get("command_id") or "")
            decision = str(review_binding.get("decision") or "").strip().upper()
            request_digest = str(review_binding.get("request_digest") or "")
            if (
                not command_id
                or len(request_digest) != 64
                or any(character not in "0123456789abcdef" for character in request_digest)
            ):
                raise ValueError("pending lifecycle review binding is incomplete")
            expected_decisions = {
                LifecycleState.CONFIRMED: {"CONFIRM", "CONFIRM_AND_APPLY"},
                LifecycleState.RESOLVED: {"CONFIRM_AND_APPLY"},
                LifecycleState.RETRYABLE: {"RETRY"},
                LifecycleState.REJECTED: {"REJECT", "CORRECT"},
                LifecycleState.EXPIRED: {"EXPIRE"},
            }.get(desired.lifecycle_state, set())
            if decision not in expected_decisions:
                raise ValueError("pending lifecycle review decision disagrees with its desired state")
            expected_history.update(
                {
                    "review_command_id": command_id,
                    "review_decision": decision,
                    "review_request_digest": request_digest,
                }
            )
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

    def _validate_pending_review_command(
        self,
        operation: ContextOperation,
        current: PendingMemoryProposal,
        review_binding: dict,
    ) -> None:
        if set(review_binding) != {"command_id", "decision", "request_digest"}:
            raise ValueError("pending lifecycle transition requires a durable review command")
        command_id = str(review_binding.get("command_id") or "")
        decision = str(review_binding.get("decision") or "").strip().upper()
        request_digest = str(review_binding.get("request_digest") or "")
        if not command_id or not decision or not request_digest:
            raise ValueError("pending lifecycle transition requires a durable review command")
        try:
            record = PendingReviewCommandStore(
                self.root,
                tenant_id=self.tenant_id,
            ).load(command_id)
        except (OSError, PendingReviewCommandIntegrityError) as exc:
            raise ValueError("pending lifecycle transition has no valid durable review command") from exc
        request = dict(record.get("request", {}) or {})
        historical = [
            dict(item)
            for item in current.lifecycle_history
            if str(dict(item).get("review_command_id") or "") == command_id
        ]
        initial_revision = (
            min(int(item.get("from_revision", 0) or 0) for item in historical)
            if historical
            else current.lifecycle_revision
        )
        if (
            record.get("status") == "failed"
            or record.get("request_digest") != request_digest
            or request.get("tenant_id") != self.tenant_id
            or request.get("owner_user_id") != operation.user_id
            or request.get("pending_uri") != current.uri
            or str(request.get("decision") or "").strip().upper() != decision
            or request.get("expected_proposal_fingerprint") != current.proposal.fingerprint
            or int(request.get("expected_lifecycle_revision", 0) or 0) != initial_revision
        ):
            raise ValueError("pending lifecycle transition conflicts with its durable review command")

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
            self._reject_control_symlink(marker, "canonical transaction receipt")
            if not marker.exists():
                raise RevisionConflictError("pending proposal cannot resolve before its canonical transaction commits")
            self._validate_transaction_marker_tenant(marker)
            diff = self._transaction_marker_diff(marker)
            self._validate_and_bind_operations(operation.user_id, diff.operations)
            committed_claims_by_key[key] = {
                str(payload.get("uri"))
                for marker_operation in diff.operations
                if marker_operation.payload.get("idempotency_key") == key
                and isinstance((payload := marker_operation.payload.get("context_object")), dict)
                and dict(payload.get("metadata", {}) or {}).get("canonical_kind") == "claim"
            }
        operation_tenant = str(operation.payload.get("tenant_id") or "default")
        for uri in claim_uris:
            claim = read_committed_canonical(self.source_store, uri, self.relation_store).object
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
                raise RevisionConflictError(
                    "pending proposal resolution Claim is not the linked committed ACTIVE Claim"
                )

    def _validate_pending_resolution_batch(self, operations: list[ContextOperation]) -> None:
        resolutions = [
            operation for operation in operations if operation.payload.get("canonical_pending_resolution") is True
        ]
        if not resolutions:
            return
        if len(resolutions) != 1:
            raise ValueError("canonical transaction can resolve exactly one pending proposal")
        resolution = resolutions[0]
        artifact_root = artifact_root_for(self.source_store)
        if artifact_root is None or not resolution.target_uri:
            raise ValueError("pending resolution has no current-head artifact root")
        confirmation_head, _confirmation_receipt, _confirmation_snapshot = load_current_head(
            artifact_root,
            resolution.target_uri,
            canonical_kind="pending_proposal",
        )
        if (
            resolution.payload.get("confirmation_receipt_digest") != confirmation_head.get("receipt_digest")
            or resolution.payload.get("confirmation_operation_id") != confirmation_head.get("current_operation_id")
            or int(resolution.payload.get("confirmation_lifecycle_revision", 0))
            != int(confirmation_head.get("current_revision", 0))
            or str(confirmation_head.get("current_lifecycle_state") or "").upper() != "CONFIRMED"
        ):
            raise ValueError("pending resolution is not bound to its current CONFIRM receipt")
        keys = {str(item) for item in resolution.payload.get("resolution_idempotency_keys", []) or [] if item}
        transaction_keys = {str(operation.payload.get("idempotency_key") or "") for operation in operations}
        claim_uris = {str(item) for item in resolution.payload.get("resolved_claim_uris", []) or [] if item}
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

    def _validate_pending_correction_batch(self, operations: list[ContextOperation]) -> None:
        corrections = [
            operation for operation in operations if operation.payload.get("canonical_pending_correction") is True
        ]
        if not corrections:
            return
        if len(corrections) != 1:
            raise ValueError("canonical transaction can correct exactly one pending proposal")
        correction = corrections[0]
        if correction.payload.get("canonical_pending_resolution") is True:
            raise ValueError("pending correction cannot also be a confirmation resolution")
        desired_payload = correction.payload.get("context_object")
        if not isinstance(desired_payload, dict):
            raise ValueError("pending correction requires a terminal pending object")
        desired = PendingMemoryProposal.from_context_object(ContextObject.from_dict(desired_payload))
        if desired.lifecycle_state != LifecycleState.REJECTED:
            raise ValueError("a corrected predecessor pending must become REJECTED")
        committed = read_committed_pending(
            self.source_store,
            str(correction.target_uri or ""),
            self.relation_store,
        )
        predecessor = PendingMemoryProposal.from_context_object(committed.object)
        if not predecessor.reason_policy.requires_new_proposal:
            raise ValueError("only a non-reviewable pending reason can use correction")
        predecessor_fingerprint = str(correction.payload.get("predecessor_proposal_fingerprint") or "")
        corrected_fingerprint = str(correction.payload.get("corrected_proposal_fingerprint") or "")
        corrected_proposal_id = str(correction.payload.get("corrected_proposal_id") or "")
        correction_task_id = str(correction.payload.get("correction_task_id") or "")
        if (
            predecessor_fingerprint != predecessor.proposal.fingerprint
            or not corrected_fingerprint
            or corrected_fingerprint == predecessor_fingerprint
            or not corrected_proposal_id
            or not correction_task_id
        ):
            raise ValueError("pending correction proposal identity is incomplete or unchanged")
        if bool(correction.payload.get("correction_requires_reextraction")) != bool(
            predecessor.reason_policy.requires_reextraction
        ):
            raise ValueError("pending correction re-extraction proof disagrees with its reason policy")
        if predecessor.reason_policy.requires_reextraction and correction_task_id == predecessor.request_identity:
            raise ValueError("fallback correction reused the predecessor extraction task")

        claim_uris = {str(item) for item in correction.payload.get("corrected_claim_uris", []) or [] if item}
        active_claims: dict[str, dict] = {}
        for operation in operations:
            if operation is correction:
                continue
            raw = operation.payload.get("context_object")
            if not isinstance(raw, dict):
                continue
            metadata = dict(raw.get("metadata", {}) or {})
            if metadata.get("canonical_kind") != "claim" or metadata.get("state") != "ACTIVE":
                continue
            current_revision = materialized_current_revision_payload(metadata)
            qualifiers = dict(current_revision.get("qualifiers", {}) or {})
            if (
                str(current_revision.get("proposal_id") or "") != corrected_proposal_id
                or str(current_revision.get("proposal_fingerprint") or "") != corrected_fingerprint
                or qualifiers.get("corrects_pending_uri") != correction.target_uri
                or qualifiers.get("corrects_pending_fingerprint") != predecessor_fingerprint
                or operation.payload.get("corrects_pending_uri") != correction.target_uri
                or operation.payload.get("corrects_pending_fingerprint") != predecessor_fingerprint
            ):
                raise ValueError("corrected Claim is not bound to its predecessor pending proposal")
            active_claims[str(raw.get("uri") or "")] = raw
        if not claim_uris or not claim_uris.issubset(active_claims):
            raise ValueError("pending correction must link an ACTIVE Claim in the same transaction")
        if any(
            str(payload.get("owner_user_id") or "") != correction.user_id
            or str(payload.get("tenant_id") or "default") != str(correction.payload.get("tenant_id") or "default")
            for payload in (active_claims[uri] for uri in claim_uris)
        ):
            raise ValueError("pending correction Claim crosses owner or tenant boundary")

    def _finalize_canonical_outbox(
        self,
        transaction_id: str,
        idempotency_key: str,
        operations: list[ContextOperation],
        *,
        slot_uri: str | None = None,
    ) -> Path:
        require_safe_path_segment(idempotency_key, "canonical idempotency_key")
        receipt_file = self._transaction_marker(idempotency_key)
        self._reject_control_symlink(receipt_file, "canonical transaction receipt")
        if not receipt_file.exists():
            raise ValueError("canonical outbox cannot commit before its immutable receipt")
        receipt = load_transaction_receipt(receipt_file)
        try:
            receipt_reference = str(receipt_file.resolve().relative_to(self.artifact_root.resolve()))
        except ValueError as exc:
            raise ValueError("canonical receipt is outside the tenant artifact root") from exc
        outbox_path = self._outbox_path(transaction_id)
        self._reject_control_symlink(outbox_path, "canonical outbox")
        outbox_complete = False
        if outbox_path.exists():
            try:
                existing = validate_outbox(
                    json.loads(outbox_path.read_text(encoding="utf-8")),
                    transaction_id=transaction_id,
                    idempotency_key=idempotency_key,
                    tenant_id=self.tenant_id,
                    user_id=operations[0].user_id,
                    operations=operations,
                )
            except (OSError, UnicodeError, json.JSONDecodeError, OutboxIntegrityError) as exc:
                raise ValueError("canonical committed outbox is unreadable") from exc
            if existing.get("status") == "committed":
                existing_operations = [
                    ContextOperation.from_dict(item)
                    for item in existing.get("operations", []) or []
                    if isinstance(item, dict)
                ]
                if operations:
                    self._validate_and_bind_operations(operations[0].user_id, existing_operations)
                if self._canonical_transaction_request_fingerprint(
                    existing_operations
                ) != self._canonical_transaction_request_fingerprint(
                    operations
                ) or self._canonical_transaction_effect_fingerprint(
                    existing_operations
                ) != self._canonical_transaction_effect_fingerprint(operations):
                    raise ValueError("canonical committed outbox conflicts with its transaction marker")
                outbox_complete = True
        if not outbox_complete:
            outbox_path = self._write_outbox_event(
                transaction_id,
                idempotency_key,
                operations,
                status="committed",
                receipt_path=receipt_reference,
                receipt_digest=str(receipt["receipt_digest"]),
            )
        # This hook proves the immutable committed outbox is durable while the
        # projection queue is still untouched.  It is intentionally emitted
        # for idempotent replay of an already-committed outbox as well.
        self._notify("after_committed_outbox", transaction_id)
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
        self._notify("after_projection_enqueue", transaction_id)
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
            if self._canonical_pending_effect(operation):
                if (
                    object_payload.get("schema_version") != PendingMemoryProposal.SCHEMA_VERSION
                    or operation.payload.get("schema_version") != PendingMemoryProposal.SCHEMA_VERSION
                    or metadata.get("canonical_kind") != "pending_proposal"
                ):
                    raise ValueError("canonical pending lifecycle effect requires a pending proposal object")
                object_tenant = str(object_payload.get("tenant_id") or "default")
                operation_tenant = str(operation.payload.get("tenant_id") or "default")
                object_owner = str(object_payload.get("owner_user_id") or operation.user_id)
                if object_tenant != operation_tenant or object_owner != operation.user_id:
                    raise ValueError("canonical pending lifecycle tenant or owner mismatch")
                scope = dict(metadata.get("scope", {}) or {})
                subject_payload = scope.get("canonical_subject")
                if not isinstance(subject_payload, dict):
                    raise ValueError("canonical pending lifecycle requires an explicit subject")
                tenants.add(object_tenant)
                owners.add(object_owner)
                slot_ids.add(str(operation.payload.get("slot_id") or ""))
                if operation.payload.get("canonical_pending_resolution") is True:
                    scope_payloads.add(json.dumps(scope, ensure_ascii=False, sort_keys=True))
                if not operation.evidence or any(
                    not item.get("event_id") or not item.get("content_hash") for item in operation.evidence
                ):
                    raise ValueError("canonical pending lifecycle effect requires durable evidence references")
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
                    current = read_committed_canonical(
                        self.source_store,
                        uri,
                        self.relation_store,
                    ).object
                    actual = int(dict(current.metadata or {}).get("revision", 0))
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                    actual = 0
                if actual != expected:
                    raise RevisionConflictError(f"revision conflict for {uri}: expected {expected}, actual {actual}")
        if len(tenants) != 1 or len(slot_ids - {""}) != 1 or len(scope_payloads) != 1:
            raise ValueError("canonical transaction must preserve tenant, slot, and scope boundaries")
        self._validate_pending_resolution_batch(operations)
        self._validate_pending_correction_batch(operations)

    def _validate_canonical_evidence(self, operation: ContextOperation) -> None:
        store = SessionArchiveStore(
            self.root,
            tenant_id=str(operation.payload.get("tenant_id") or "default"),
        )
        verified_sources: set[str] = set()
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
                # The final-state validator distinguishes changed fields
                # (which require this transaction's evidence) from unchanged
                # fields (which must retain prior provenance).  Requiring all
                # materialized field refs here would make immutable provenance
                # impossible across revisions.
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
            slot = read_committed_canonical(
                self.source_store,
                slot_uri,
                self.relation_store,
            ).object
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return
        metadata = dict(slot.metadata or {})
        claim_ids = [str(item) for item in metadata.get("claim_ids", []) or []]
        active: list[str] = []
        for claim_id in claim_ids:
            try:
                claim = read_committed_canonical(
                    self.source_store,
                    f"{slot_uri}/claims/{claim_id}",
                    self.relation_store,
                ).object
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
        # RelationStore publication is a separate, redo-proved phase.  Writing
        # relations here would make ``after_source_effect`` indistinguishable
        # from ``after_relation_effect`` after a process crash.

    def _build_canonical_relation_manifest(
        self,
        operation: ContextOperation,
        before_object: ContextObject | None,
    ) -> dict:
        payload = operation.payload.get("context_object")
        if not isinstance(payload, dict):
            raise ValueError("canonical relation manifest requires context_object")
        desired = ContextObject.from_dict(payload)
        expected = self._canonical_relation_specs(operation, desired)
        expected_keys = {self._relation_spec_key(spec) for spec in expected}
        previous_keys = self._canonical_managed_relation_keys(before_object) if self.relation_store is not None else []
        remove = self._unique_relation_keys(
            [key for key in previous_keys if self._relation_spec_key(key) not in expected_keys]
        )
        core = {
            "schema_version": "canonical_relation_manifest_v1",
            "operation_id": operation.operation_id,
            "user_id": operation.user_id,
            "tenant_id": str(operation.payload.get("tenant_id") or "default"),
            "transaction_id": str(operation.payload.get("transaction_id") or ""),
            "idempotency_key": str(operation.payload.get("idempotency_key") or ""),
            "target_uri": operation.target_uri,
            "expected": expected,
            "remove": remove,
        }
        return {**core, "fingerprint": stable_hash(core, length=64)}

    def _validate_canonical_relation_manifest(
        self,
        operation: ContextOperation,
        manifest: dict,
    ) -> None:
        if manifest.get("schema_version") != "canonical_relation_manifest_v1":
            raise RedoIntegrityError("canonical relation manifest schema is unsupported")
        core = {key: value for key, value in manifest.items() if key != "fingerprint"}
        if manifest.get("fingerprint") != stable_hash(core, length=64):
            raise RedoIntegrityError("canonical relation manifest fingerprint is corrupt")
        if (
            manifest.get("operation_id") != operation.operation_id
            or manifest.get("user_id") != operation.user_id
            or manifest.get("tenant_id") != str(operation.payload.get("tenant_id") or "default")
            or manifest.get("transaction_id") != str(operation.payload.get("transaction_id") or "")
            or manifest.get("idempotency_key") != str(operation.payload.get("idempotency_key") or "")
            or manifest.get("target_uri") != operation.target_uri
            or not isinstance(manifest.get("expected"), list)
            or not isinstance(manifest.get("remove"), list)
        ):
            raise RedoIntegrityError("canonical relation manifest crosses its operation boundary")

    def _apply_canonical_relation_manifest(
        self,
        operation: ContextOperation,
        manifest: dict,
    ) -> None:
        self._validate_canonical_relation_manifest(operation, manifest)
        if self.relation_store is None:
            if manifest.get("expected") or manifest.get("remove"):
                raise RedoIntegrityError("canonical relation manifest requires a RelationStore")
            return
        for key in manifest.get("remove", []) or []:
            self.relation_store.delete_relation(
                str(key["source_uri"]),
                str(key["relation_type"]),
                str(key["target_uri"]),
            )
        self._ensure_relation_specs([dict(item) for item in manifest.get("expected", []) or []])
        self._validate_canonical_relation_manifest_effect(manifest)

    def _validate_canonical_relation_manifest_effect(self, manifest: dict) -> None:
        if self.relation_store is None:
            if manifest.get("expected") or manifest.get("remove"):
                raise RedoIntegrityError("canonical relation effect has no RelationStore")
            return
        for spec in manifest.get("expected", []) or []:
            actual = {
                canonical_json(self._relation_effect_spec(relation))
                for relation in self.relation_store.relations_of(str(spec["source_uri"]))
            }
            if canonical_json(spec) not in actual:
                raise RedoIntegrityError("canonical RelationStore effect is incomplete")
        for key in manifest.get("remove", []) or []:
            if any(
                relation.source_uri == key["source_uri"]
                and relation.relation_type == key["relation_type"]
                and relation.target_uri == key["target_uri"]
                for relation in self.relation_store.relations_of(str(key["source_uri"]))
            ):
                raise RedoIntegrityError("canonical RelationStore retained a removed managed relation")

    def _canonical_relation_specs(
        self,
        operation: ContextOperation,
        obj: ContextObject,
    ) -> list[dict]:
        if self.relation_store is None:
            return []
        metadata = dict(obj.metadata or {})
        relation_metadata = {
            "tenant_id": obj.tenant_id or "default",
            "owner_user_id": obj.owner_user_id,
            "canonical_transaction_id": operation.payload.get("transaction_id"),
            "canonical_idempotency_key": operation.payload.get("idempotency_key"),
            "source_revision": metadata.get("revision"),
            "commit_group_id": operation.payload.get("commit_group_id"),
        }
        specs = []
        for relation in obj.relations:
            specs.append(
                self._relation_spec(
                    relation.source_uri,
                    relation.relation_type,
                    relation.target_uri,
                    {**dict(relation.metadata or {}), **relation_metadata},
                    weight=relation.weight,
                )
            )
        kind = str(metadata.get("canonical_kind") or "")
        if kind == "claim":
            slot_uri = obj.uri.rsplit("/claims/", 1)[0]
            specs.append(self._relation_spec(obj.uri, "belongs_to_slot", slot_uri, relation_metadata))
        elif kind == "slot":
            specs.extend(
                self._relation_spec(
                    obj.uri,
                    "has_claim",
                    f"{obj.uri}/claims/{claim_id}",
                    relation_metadata,
                )
                for claim_id in sorted(str(item) for item in metadata.get("claim_ids", []) or [] if str(item))
            )
        return self._unique_relation_specs(specs)

    def _canonical_managed_relation_keys(
        self,
        obj: ContextObject | None,
    ) -> list[dict]:
        if obj is None:
            return []
        keys = [self._relation_key_payload(self._relation_effect_spec(relation)) for relation in obj.relations]
        metadata = dict(obj.metadata or {})
        kind = str(metadata.get("canonical_kind") or "")
        if kind == "claim":
            slot_uri = obj.uri.rsplit("/claims/", 1)[0]
            keys.append(self._relation_key_payload(self._relation_spec(obj.uri, "belongs_to_slot", slot_uri, {})))
        elif kind == "slot":
            keys.extend(
                self._relation_key_payload(
                    self._relation_spec(
                        obj.uri,
                        "has_claim",
                        f"{obj.uri}/claims/{claim_id}",
                        {},
                    )
                )
                for claim_id in sorted(str(item) for item in metadata.get("claim_ids", []) or [] if str(item))
            )
        return self._unique_relation_keys(keys)

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

    def _capture_canonical_source_effect(
        self,
        operation: ContextOperation,
        relation_manifest: dict,
    ) -> dict:
        self._validate_canonical_relation_manifest(operation, relation_manifest)
        self._validate_existing_canonical_effect(operation)
        self._validate_canonical_relation_manifest_effect(relation_manifest)
        planned = planned_effect_manifest(operation, relation_manifest)
        core = {
            "schema_version": "canonical_source_effect_v1",
            "operation_id": operation.operation_id,
            "transaction_id": str(operation.payload.get("transaction_id") or ""),
            "idempotency_key": str(operation.payload.get("idempotency_key") or ""),
            "tenant_id": self.tenant_id,
            "user_id": operation.user_id,
            "uri": planned["uri"],
            "object_digest": planned["object_digest"],
            "content_digest": planned["content_digest"],
            "revision": planned["revision"],
            "relation_manifest_digest": planned["relation_manifest_digest"],
            "planned_effect_digest": planned["effect_digest"],
        }
        return {**core, "effect_digest": canonical_digest(core)}

    def _validate_canonical_source_effect(
        self,
        operation: ContextOperation,
        source_effect: dict | None,
        relation_manifest: dict | None,
    ) -> None:
        if not isinstance(source_effect, dict) or not isinstance(relation_manifest, dict):
            raise RedoIntegrityError("canonical redo is missing its Source or Relation effect")
        stored_core = {key: value for key, value in source_effect.items() if key != "effect_digest"}
        if source_effect.get("schema_version") != "canonical_source_effect_v1" or source_effect.get(
            "effect_digest"
        ) != canonical_digest(stored_core):
            raise RedoIntegrityError("canonical redo Source effect digest is corrupt")
        try:
            actual = self._capture_canonical_source_effect(operation, relation_manifest)
        except (FileNotFoundError, RevisionConflictError, ValueError) as exc:
            raise RedoIntegrityError("canonical redo Source effect does not match durable state") from exc
        if actual != source_effect:
            raise RedoIntegrityError("canonical redo Source effect does not match durable state")

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
        require_safe_path_segment(idempotency_key, "canonical idempotency_key")
        path = self._outbox_path(transaction_id)
        self._reject_control_symlink(path, "canonical outbox")
        if not operations:
            raise ValueError("canonical outbox requires transaction operations")
        self._ensure_canonical_planning_digest(operations)
        self._validate_and_bind_operations(operations[0].user_id, operations)
        claim_revisions: list[dict] = []
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
        existing: dict | None = None
        if path.exists():
            try:
                existing_payload = json.loads(path.read_text(encoding="utf-8"))
                existing = validate_outbox(
                    existing_payload,
                    transaction_id=transaction_id,
                    idempotency_key=idempotency_key,
                    tenant_id=self.tenant_id,
                    user_id=operations[0].user_id,
                    operations=operations,
                )
            except (OSError, UnicodeError, json.JSONDecodeError, OutboxIntegrityError) as exc:
                raise ValueError("canonical outbox is corrupt or crosses its transaction boundary") from exc
            assert_transition(str(existing["status"]), status)
        before_payloads = (
            [self._before_image_payload(item) for item in before_images]
            if before_images is not None
            else list((existing or {}).get("before_images", []) or [])
        )
        if relation_manifests is not None:
            effects = [
                planned_effect_manifest(operation, relation_manifests.get(operation.operation_id))
                for operation in operations
            ]
        elif existing is not None:
            effects = list(existing.get("effect_manifests", []) or [])
        else:
            effects = [planned_effect_manifest(operation, None) for operation in operations]
        event = build_outbox(
            transaction_id=transaction_id,
            idempotency_key=idempotency_key,
            tenant_id=self.tenant_id,
            user_id=operations[0].user_id,
            operations=operations,
            status=status,
            before_images=before_payloads,
            effect_manifests=effects,
            claim_revisions=claim_revisions,
            commit_group_id=next(
                (
                    str(operation.payload.get("commit_group_id"))
                    for operation in operations
                    if operation.payload.get("commit_group_id")
                ),
                "",
            ),
            receipt_path=receipt_path,
            receipt_digest=receipt_digest,
        )
        # The outbox is a durable transaction boundary, not merely an
        # internal serialization detail.  Re-validate the fully assembled
        # envelope before publication so a builder regression cannot persist
        # a Claim projection set, prepared intent, or receipt binding that is
        # detached from the immutable operation set.
        try:
            event = validate_outbox(
                event,
                transaction_id=transaction_id,
                idempotency_key=idempotency_key,
                tenant_id=self.tenant_id,
                user_id=operations[0].user_id,
                operations=operations,
                allowed_statuses={status},
            )
        except OutboxIntegrityError as exc:
            raise ValueError("canonical outbox failed pre-publication validation") from exc
        try:
            if status == "prepared":
                immutable_intent = self.planning_proofs.ensure_canonical_intent(
                    event,
                    operations=operations,
                )
            else:
                immutable_intent = self.planning_proofs.load_canonical_intent(
                    transaction_id,
                    operations=operations,
                    prepared_intent_digest=str(event["prepared_intent_digest"]),
                )
        except PlanningProofIntegrityError as exc:
            raise ValueError("canonical outbox transition is detached from its immutable prepared intent") from exc
        if immutable_intent["prepared_intent_digest"] != event["prepared_intent_digest"]:
            raise ValueError("canonical outbox prepared intent digest changed across transition")
        self._reject_control_symlink(path, "canonical outbox")
        atomic_write_json(path, event, artifact_root=self.artifact_root)
        return path

    def _before_image_payload(self, snapshot: dict) -> dict:
        obj = snapshot.get("object")
        relations = sorted(
            (
                self._relation_effect_spec(relation)
                for relation in snapshot.get("relations", []) or []
                if isinstance(relation, ContextRelation)
            ),
            key=canonical_json,
        )
        return {
            "uri": str(snapshot.get("uri", "")),
            "exists": bool(snapshot.get("exists")),
            "object": obj.to_dict() if isinstance(obj, ContextObject) else None,
            "content": str(snapshot.get("content", "")),
            "relations": relations,
            "relations_digest": canonical_digest(relations),
        }

    def _capture_canonical_state(self, operations: list[ContextOperation]) -> list[dict]:
        snapshots = []
        for operation in operations:
            payload = operation.payload.get("context_object")
            if not isinstance(payload, dict):
                continue
            uri = str(payload["uri"])
            try:
                committed = read_committed_canonical(
                    self.source_store,
                    uri,
                    self.relation_store,
                )
                obj = committed.object
                exists = True
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                obj = None
                exists = False
            if obj is not None:
                content = committed_content(committed)
            else:
                content = ""
            relations = list(committed_relations(committed)) if obj is not None else []
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
        transaction_id = require_safe_path_segment(transaction_id, "canonical transaction_id")
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
        key = require_safe_path_segment(idempotency_key, "canonical idempotency_key")
        return self.artifact_root / "system" / "transactions" / f"{key}.json"

    def _outbox_path(self, transaction_id: str) -> Path:
        key = require_safe_path_segment(transaction_id, "canonical transaction_id")
        return self.artifact_root / "system" / "outbox" / f"{key}.json"

    def _ensure_canonical_transaction_diff(
        self,
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
        path = self.artifact_root / "system" / "diffs" / f"diff_{transaction_key}.json"
        self._reject_control_symlink(path, "canonical transaction diff")
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("canonical transaction diff must be a JSON object")
                diff = self._diff_from_payload(payload)
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
                or self._canonical_transaction_request_fingerprint(diff.operations)
                != self._canonical_transaction_request_fingerprint(operations)
                or self._canonical_transaction_effect_fingerprint(diff.operations)
                != self._canonical_transaction_effect_fingerprint(operations)
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
        self.diff_writer.write(diff)
        return diff

    @staticmethod
    def _reject_control_symlink(path: Path, label: str) -> None:
        if path.is_symlink():
            raise ValueError(f"{label} cannot be a symbolic link")

    def _write_transaction_marker(
        self,
        path: Path,
        diff: ContextDiff,
        operations: list[ContextOperation],
        *,
        relation_manifests: dict[str, dict] | None = None,
    ) -> None:
        if not operations:
            raise ValueError("canonical transaction marker requires operations")
        keys = {self._validate_canonical_artifact_keys(operation)[1] for operation in operations}
        if len(keys) != 1 or path != self._transaction_marker(next(iter(keys))):
            raise ValueError("canonical transaction marker path does not match its operations")
        self._reject_control_symlink(path, "canonical transaction receipt")
        if path.exists():
            self._validate_transaction_marker(path, operations)
            return
        transaction_ids = {self._validate_canonical_artifact_keys(operation)[0] for operation in operations}
        if len(transaction_ids) != 1:
            raise ValueError("canonical transaction marker requires one transaction id")
        relation_effects = self._marker_relation_effects(relation_manifests)
        outbox_path = self._outbox_path(next(iter(transaction_ids)))
        self._reject_control_symlink(outbox_path, "canonical outbox")
        if not outbox_path.exists():
            raise ValueError("canonical receipt requires its previously published prepared outbox intent")
        try:
            outbox = validate_outbox(
                json.loads(outbox_path.read_text(encoding="utf-8")),
                transaction_id=next(iter(transaction_ids)),
                idempotency_key=next(iter(keys)),
                tenant_id=self.tenant_id,
                user_id=operations[0].user_id,
                operations=operations,
                allowed_statuses={"prepared", "source_committed"},
            )
            immutable_intent = self.planning_proofs.load_canonical_intent(
                next(iter(transaction_ids)),
                operations=operations,
                prepared_intent_digest=str(outbox["prepared_intent_digest"]),
            )
            intent_digest = str(immutable_intent["prepared_intent_digest"])
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            OutboxIntegrityError,
            PlanningProofIntegrityError,
        ) as exc:
            raise ValueError("canonical receipt requires a valid prepared intent") from exc
        planning_digests = {str(operation.payload.get("planning_digest") or "") for operation in operations}
        if len(planning_digests) != 1 or "" in planning_digests:
            raise ValueError("canonical receipt requires exactly one planning digest")
        payload = build_transaction_receipt(
            transaction_id=next(iter(transaction_ids)),
            idempotency_key=next(iter(keys)),
            tenant_id=self.tenant_id,
            user_id=operations[0].user_id,
            commit_group_id=next(
                (
                    str(operation.payload.get("commit_group_id") or "")
                    for operation in operations
                    if operation.payload.get("commit_group_id")
                ),
                "",
            ),
            operations=operations,
            diff=diff.to_dict(),
            planning_digest=next(iter(planning_digests)),
            prepared_intent_digest=intent_digest,
            prepared_intent_schema_version=CANONICAL_PREPARED_INTENT_SCHEMA_VERSION,
            relation_effects=relation_effects,
            created_at=diff.created_at,
        )
        self._reject_control_symlink(path, "canonical transaction receipt")
        atomic_create_json(path, payload, artifact_root=self.artifact_root)

    def _validate_transaction_marker(
        self,
        path: Path,
        operations: list[ContextOperation],
    ) -> ContextDiff:
        if not operations:
            raise ValueError("canonical transaction marker validation requires operations")
        keys = {self._validate_canonical_artifact_keys(operation)[1] for operation in operations}
        if len(keys) != 1 or path != self._transaction_marker(next(iter(keys))):
            raise ValueError("canonical transaction marker path does not match its operations")
        transaction_ids = {self._validate_canonical_artifact_keys(operation)[0] for operation in operations}
        if len(transaction_ids) != 1:
            raise ValueError("canonical transaction marker requires one transaction id")
        try:
            payload = validate_marker(
                path,
                self.source_store,
                self.relation_store,
                transaction_id=next(iter(transaction_ids)),
                idempotency_key=next(iter(keys)),
                tenant_id=self.tenant_id,
                user_id=operations[0].user_id,
                operation_ids=[operation.operation_id for operation in operations],
            )
        except EffectProofError as exc:
            if path.exists():
                quarantine_control_file(
                    self.artifact_root,
                    path,
                    kind="transaction_marker",
                    error=exc,
                    identifiers={
                        "transaction_id": next(iter(transaction_ids)),
                        "idempotency_key": next(iter(keys)),
                    },
                )
            raise ValueError("canonical transaction marker cannot prove its durable effect") from exc
        diff_payload = payload.get("diff")
        if not isinstance(diff_payload, dict):
            raise ValueError("canonical transaction marker is missing its persisted diff")
        diff = self._diff_from_payload(diff_payload)
        self._validate_and_bind_operations(operations[0].user_id, operations)
        self._validate_and_bind_operations(operations[0].user_id, diff.operations)
        if diff.user_id != operations[0].user_id:
            raise ValueError("canonical transaction marker crosses a user boundary")
        if self._canonical_transaction_request_fingerprint(
            diff.operations
        ) != self._canonical_transaction_request_fingerprint(
            operations
        ) or self._canonical_transaction_effect_fingerprint(
            diff.operations
        ) != self._canonical_transaction_effect_fingerprint(operations):
            raise ValueError("canonical idempotency marker conflicts with the requested transaction")
        return diff

    def _validate_transaction_marker_tenant(self, path: Path) -> None:
        if path.is_symlink():
            raise ValueError("canonical transaction receipt cannot be a symbolic link")
        payload = json.loads(path.read_text(encoding="utf-8"))
        tenant = self._validate_tenant_id(payload["tenant_id"], "canonical transaction marker tenant_id")
        if tenant != self.tenant_id:
            raise ValueError("canonical transaction marker crosses the bound tenant")

    def _transaction_marker_diff(self, path: Path) -> ContextDiff:
        if path.is_symlink():
            raise ValueError("canonical transaction receipt cannot be a symbolic link")
        payload = json.loads(path.read_text(encoding="utf-8"))
        diff_payload = payload.get("diff")
        if not isinstance(diff_payload, dict):
            raise ValueError("canonical transaction marker is missing its persisted diff")
        operations_payload = payload.get("operations")
        if payload.get("schema_version") not in {
            "effect_marker_v1",
            TRANSACTION_RECEIPT_SCHEMA_VERSION,
        } or not isinstance(operations_payload, list):
            raise ValueError("canonical transaction marker schema is unsupported")
        if payload.get("schema_version") == TRANSACTION_RECEIPT_SCHEMA_VERSION:
            try:
                validate_transaction_receipt(payload)
            except ReceiptIntegrityError as exc:
                raise ValueError("canonical transaction receipt is corrupt") from exc
        return self._diff_from_payload(diff_payload)

    def _marker_relation_effects(
        self,
        relation_manifests: dict[str, dict] | None,
    ) -> list[dict]:
        if not relation_manifests:
            return []
        by_identity: dict[str, dict] = {}
        for operation_id in sorted(relation_manifests):
            for effect in relation_effects_from_manifest(relation_manifests[operation_id]):
                identity_key = canonical_json(effect["identity"])
                current = by_identity.get(identity_key)
                if current is None or effect["expected_exists"] is True:
                    by_identity[identity_key] = effect
        return [by_identity[key] for key in sorted(by_identity)]

    def _canonical_transaction_request_fingerprint(self, operations: list[ContextOperation]) -> str:
        normalized = []
        for operation in sorted(operations, key=lambda item: item.operation_id):
            payload = json.loads(json.dumps(operation.to_dict(), ensure_ascii=False))
            payload.pop("status", None)
            payload.pop("created_at", None)
            self._strip_relation_timestamps(payload)
            normalized.append(payload)
        canonical_json(normalized)
        return stable_hash(normalized, length=64)

    def _canonical_transaction_request_fingerprint_v2(self, operations: list[ContextOperation]) -> str:
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
                    "context_object": self._context_object_without_relation_timestamps(
                        operation.payload.get("context_object")
                    ),
                    "content": operation.payload.get("content", ""),
                }
            )
        canonical_json(effects)
        return stable_hash(effects, length=64)

    def _canonical_transaction_effect_fingerprint_v2(self, operations: list[ContextOperation]) -> str:
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

    def _strip_relation_timestamps(self, operation_payload: dict) -> None:
        payload = operation_payload.get("payload")
        if not isinstance(payload, dict):
            return
        context_object = payload.get("context_object")
        payload["context_object"] = self._context_object_without_relation_timestamps(context_object)

    def _context_object_without_relation_timestamps(self, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = json.loads(json.dumps(value, ensure_ascii=False))
        relations = normalized.get("relations")
        if isinstance(relations, list):
            for relation in relations:
                if isinstance(relation, dict):
                    relation.pop("created_at", None)
        return normalized

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

    def resume(
        self,
        user_id: str,
        operation: ContextOperation,
        phase: str,
        *,
        source_effect: dict | None = None,
        relation_manifest: dict | None = None,
    ) -> bool:
        """处理 resume 这一步。"""

        self._validate_redo_boundary(
            user_id,
            operation,
            source_effect=source_effect,
            relation_manifest=relation_manifest,
        )
        if operation.payload.get("canonical_memory") is True:
            transaction_id = str(operation.payload.get("transaction_id") or "")
            entries = [
                entry
                for entry in self.redo.pending_entries()
                if entry.operation.user_id == user_id
                and str(entry.operation.payload.get("transaction_id") or "") == transaction_id
            ]
            if not entries:
                raise RedoIntegrityError("canonical recovery requires the complete durable transaction batch")
            return operation.operation_id in self.resume_canonical_batch(user_id, entries)
        if phase in {"started", "begin"}:
            return self._resume_started_source_effect(
                user_id,
                operation,
                relation_manifest=relation_manifest,
            )
        with ExitStack() as locks:
            guards = [
                locks.enter_context(self.path_lock.acquire(self._lock_key(lock_key)))
                for lock_key in self._regular_lock_keys(operation)
            ]
            guard = guards[0]
            with self.path_lock.fenced(guards):
                return self._resume_under_guard(
                    user_id,
                    operation,
                    phase,
                    source_effect=source_effect,
                    relation_manifest=relation_manifest,
                    guard=guard,
                )

    def _resume_started_source_effect(
        self,
        user_id: str,
        operation: ContextOperation,
        *,
        relation_manifest: dict | None,
    ) -> bool:
        """Adopt a fully matching Source effect from the begin -> phase crash window."""

        with ExitStack() as locks:
            guards = [
                locks.enter_context(self.path_lock.acquire(self._lock_key(lock_key)))
                for lock_key in self._regular_lock_keys(operation)
            ]
            guard = guards[0]
            with self.path_lock.fenced(guards):
                if relation_manifest is not None:
                    self._validate_regular_relation_manifest(operation, relation_manifest)
                elif self.relation_store is not None:
                    raise RedoIntegrityError("regular redo entry is missing its relation manifest")
                replay_source = operation.action == OperationAction.REFRESH_LAYERS
                if not replay_source:
                    effect = self._capture_regular_source_effect(operation, relation_manifest)
                    try:
                        self._validate_regular_recovery_effect(
                            user_id,
                            operation,
                            effect,
                            require_relation_presence=False,
                            relation_manifest=relation_manifest,
                        )
                    except RedoIntegrityError:
                        replay_source = True
                if replay_source:
                    self._apply_source(operation)
                if isinstance(relation_manifest, dict):
                    self._apply_regular_relation_manifest(operation, relation_manifest)
                effect = self._capture_regular_source_effect(operation, relation_manifest)
                self._validate_regular_recovery_effect(
                    user_id,
                    operation,
                    effect,
                    relation_manifest=relation_manifest,
                )
                self.redo.advance(
                    operation,
                    phase="source_written",
                    source_effect=effect,
                    relation_manifest=relation_manifest,
                )
                return self._resume_under_guard(
                    user_id,
                    operation,
                    "source_written",
                    source_effect=effect,
                    relation_manifest=relation_manifest,
                    guard=guard,
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
        del guard
        if phase == "head_published":
            if operation.payload.get("canonical_pending_proposal") is not True:
                raise RedoIntegrityError("regular head-published redo is not a pending lifecycle operation")
            marker = self._operation_marker(operation.operation_id)
            self._reject_control_symlink(marker, "pending operation receipt")
            try:
                receipt = load_transaction_receipt(marker)
            except ReceiptIntegrityError as exc:
                raise RedoIntegrityError("head-published pending redo has no valid immutable receipt") from exc
            self._validate_operation_marker(marker, operation)
            self._validate_head_published_receipt(marker, receipt)
            self.redo.commit(operation.operation_id)
            return False
        if phase in {"committed"}:
            if operation.payload.get("canonical_memory") is not True:
                marker = self._operation_marker(operation.operation_id)
                if not marker.exists():
                    raise RedoIntegrityError("committed redo entry has no operation marker")
                stored = self._validate_operation_marker(marker, operation)
                if stored.payload.get("canonical_pending_proposal") is True:
                    try:
                        receipt = load_transaction_receipt(marker)
                    except ReceiptIntegrityError as exc:
                        raise RedoIntegrityError(
                            "committed pending redo has no valid immutable receipt"
                        ) from exc
                    # ``committed`` is a legacy post-publication redo phase.
                    # It cannot authorize publication of a missing lifecycle
                    # head: doing so would turn historical receipt replay into
                    # mutable current-state repair.  Only an explicitly
                    # pre-head phase may complete publication.
                    self._validate_head_published_receipt(marker, receipt)
            self.redo.commit(operation.operation_id)
            return False
        self._validate_and_restore_regular_recovery_effect(
            user_id,
            operation,
            source_effect,
            relation_manifest,
        )
        if phase == "source_written":
            self._apply_index(operation)
            self.redo.advance(operation, phase="index_written")
            self.audit.record(user_id, "context_operation_committed", operation.to_dict())
            self.redo.advance(operation, phase="audit_written")
            self._finalize_single_regular_operation(
                user_id,
                operation,
                source_effect=source_effect,
                relation_manifest=relation_manifest,
            )
            return True
        if phase == "index_written":
            self.audit.record(user_id, "context_operation_committed", operation.to_dict())
            self.redo.advance(operation, phase="audit_written")
            self._finalize_single_regular_operation(
                user_id,
                operation,
                source_effect=source_effect,
                relation_manifest=relation_manifest,
            )
            return True
        if phase == "audit_written":
            self._finalize_single_regular_operation(
                user_id,
                operation,
                source_effect=source_effect,
                relation_manifest=relation_manifest,
            )
            return True
        if phase == "diff_written":
            diff = self._ensure_single_operation_diff(user_id, operation)
            self._write_operation_marker(
                operation,
                source_effect=source_effect,
                relation_manifest=relation_manifest,
                diff=diff,
            )
            self.redo.commit(operation.operation_id)
            return True
        return False

    def _capture_regular_source_effect(
        self,
        operation: ContextOperation,
        relation_manifest: dict | None = None,
    ) -> dict:
        uris = self._regular_source_effect_uris(operation)
        snapshots = []
        for uri in uris:
            try:
                obj = self.source_store.read_object(uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                snapshots.append({"uri": uri, "exists": False})
                continue
            layer_hashes: dict[str, str | None] = {}
            layer_uris = tuple(
                dict.fromkeys(
                    item for item in (obj.layers.l0_uri, obj.layers.l1_uri, obj.layers.l2_uri or obj.uri) if item
                )
            )
            for layer_uri in layer_uris:
                try:
                    content = self.source_store.read_content(str(layer_uri))
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                    layer_hashes[str(layer_uri)] = None
                else:
                    layer_hashes[str(layer_uri)] = evidence_hash(content)
            snapshots.append(
                {
                    "uri": uri,
                    "exists": True,
                    "object": obj.to_dict(),
                    "layer_hashes": layer_hashes,
                }
            )
        core = {
            "schema_version": "regular_source_effect_v2",
            "operation_id": operation.operation_id,
            "user_id": operation.user_id,
            "uris": uris,
            "snapshots": snapshots,
            "relations": (
                list(relation_manifest.get("expected", []) or [])
                if isinstance(relation_manifest, dict)
                else self._expected_regular_relation_specs(operation)
            ),
            "relation_manifest_fingerprint": (
                str(relation_manifest.get("fingerprint") or "") if isinstance(relation_manifest, dict) else ""
            ),
        }
        return {**core, "fingerprint": stable_hash(core, length=64)}

    def _validate_regular_recovery_effect(
        self,
        user_id: str,
        operation: ContextOperation,
        source_effect: dict | None,
        *,
        require_relation_presence: bool = True,
        relation_manifest: dict | None = None,
    ) -> None:
        if not isinstance(source_effect, dict):
            raise RedoIntegrityError("regular redo entry is missing its SourceStore effect")
        if source_effect.get("schema_version") != "regular_source_effect_v2":
            raise RedoIntegrityError("regular redo SourceStore effect schema is unsupported")
        stored_core = {key: value for key, value in source_effect.items() if key != "fingerprint"}
        if source_effect.get("fingerprint") != stable_hash(stored_core, length=64):
            raise RedoIntegrityError("regular redo SourceStore effect fingerprint is corrupt")
        if (
            source_effect.get("operation_id") != operation.operation_id
            or source_effect.get("user_id") != user_id
            or list(source_effect.get("uris", []) or []) != self._regular_source_effect_uris(operation)
        ):
            raise RedoIntegrityError("regular redo SourceStore effect is bound to another operation")
        actual = self._capture_regular_source_effect(operation, relation_manifest)
        if actual.get("fingerprint") != source_effect.get("fingerprint"):
            raise RedoIntegrityError("regular redo SourceStore effect does not match durable state")
        expected_tenant = self._regular_operation_tenant(operation)
        self._validate_regular_action_postcondition(operation, actual)
        if relation_manifest is not None:
            self._validate_regular_relation_manifest(operation, relation_manifest)
        elif self.relation_store is not None:
            raise RedoIntegrityError("regular redo entry is missing its relation manifest")
        expected_relations = (
            list(relation_manifest.get("expected", []) or [])
            if isinstance(relation_manifest, dict)
            else self._expected_regular_relation_specs(operation)
        )
        if source_effect.get("relations") != expected_relations:
            raise RedoIntegrityError("regular redo relation effect does not match its operation")
        if source_effect.get("relation_manifest_fingerprint", "") != (
            str(relation_manifest.get("fingerprint") or "") if isinstance(relation_manifest, dict) else ""
        ):
            raise RedoIntegrityError("regular redo relation manifest does not match its SourceStore effect")
        for snapshot in actual.get("snapshots", []) or []:
            if not snapshot.get("exists") or not isinstance(snapshot.get("object"), dict):
                raise RedoIntegrityError("regular redo SourceStore effect is missing its target object")
            obj = ContextObject.from_dict(snapshot["object"])
            try:
                parsed = ContextURI.parse(obj.uri)
            except (TypeError, ValueError) as exc:
                raise RedoIntegrityError("regular redo SourceStore URI is invalid") from exc
            if parsed.authority == "user":
                if parsed.user_id != user_id or obj.owner_user_id != user_id:
                    raise RedoIntegrityError("regular redo SourceStore effect crosses a user boundary")
            elif obj.owner_user_id not in {None, "", user_id}:
                raise RedoIntegrityError("regular redo SourceStore effect crosses an owner boundary")
            if str(obj.tenant_id or "default") != expected_tenant:
                raise RedoIntegrityError("regular redo SourceStore effect crosses a tenant boundary")
            if obj.context_type != operation.context_type:
                raise RedoIntegrityError("regular redo SourceStore effect changes context type")
        if require_relation_presence:
            if isinstance(relation_manifest, dict):
                self._validate_regular_relation_manifest_effect(relation_manifest)
            else:
                self._validate_regular_relation_postcondition(expected_relations)

    def _validate_and_restore_regular_recovery_effect(
        self,
        user_id: str,
        operation: ContextOperation,
        source_effect: dict | None,
        relation_manifest: dict | None,
    ) -> None:
        self._validate_regular_recovery_effect(
            user_id,
            operation,
            source_effect,
            require_relation_presence=False,
            relation_manifest=relation_manifest,
        )
        if isinstance(relation_manifest, dict):
            self._apply_regular_relation_manifest(operation, relation_manifest)
        else:
            assert isinstance(source_effect, dict)
            self._restore_regular_relation_effect(operation, source_effect)
        self._validate_regular_recovery_effect(
            user_id,
            operation,
            source_effect,
            relation_manifest=relation_manifest,
        )

    def _validate_regular_action_postcondition(
        self,
        operation: ContextOperation,
        effect: dict,
    ) -> None:
        snapshots = {
            str(snapshot.get("uri") or ""): snapshot
            for snapshot in effect.get("snapshots", []) or []
            if isinstance(snapshot, dict)
        }

        def required(uri: str) -> tuple[ContextObject, dict]:
            snapshot = snapshots.get(uri)
            if snapshot is None or not snapshot.get("exists") or not isinstance(snapshot.get("object"), dict):
                raise RedoIntegrityError(f"regular redo {operation.action.value} effect is missing {uri}")
            return ContextObject.from_dict(snapshot["object"]), snapshot

        object_payload = operation.payload.get("context_object")
        desired = ContextObject.from_dict(object_payload) if isinstance(object_payload, dict) else None
        if operation.action in {OperationAction.ADD, OperationAction.UPDATE, OperationAction.MERGE}:
            if desired is None:
                raise RedoIntegrityError("regular object write has no desired object")
            actual, snapshot = required(desired.uri)
            normalized = self._normalized_regular_object_effect(operation)
            if not isinstance(normalized, dict) or canonical_json(actual.to_dict()) != canonical_json(normalized):
                raise RedoIntegrityError("regular object write did not persist its desired object")
            content = str(operation.payload.get("content", ""))
            if content:
                content_uri = actual.layers.l2_uri or actual.uri
                if dict(snapshot.get("layer_hashes", {}) or {}).get(content_uri) != evidence_hash(content):
                    raise RedoIntegrityError("regular object write did not persist its desired content")
            return
        if operation.action == OperationAction.SUPERSEDE:
            if not operation.target_uri or desired is None:
                raise RedoIntegrityError("supersede effect is missing an old or replacement URI")
            old, _old_snapshot = required(operation.target_uri)
            new, new_snapshot = required(desired.uri)
            reason = str(operation.payload.get("reason") or operation.payload.get("supersede_reason") or "")
            superseded_at = str(new.metadata.get("superseded_at") or "")
            expected_new = ContextObject.from_dict(desired.to_dict())
            expected_new.lifecycle_state = LifecycleState.ACTIVE
            expected_new.metadata = {
                **expected_new.metadata,
                "supersedes": old.uri,
                "superseded_at": superseded_at,
                "supersede_reason": reason,
            }
            content = str(operation.payload.get("content", ""))
            content_uri = new.layers.l2_uri or new.uri
            if (
                old.lifecycle_state != LifecycleState.OBSOLETE
                or str(old.metadata.get("superseded_by") or "") != new.uri
                or str(old.metadata.get("supersede_reason") or "") != reason
                or not superseded_at
                or str(old.metadata.get("superseded_at") or "") != superseded_at
                or new.lifecycle_state != LifecycleState.ACTIVE
                or str(new.metadata.get("supersedes") or "") != old.uri
                or str(new.metadata.get("supersede_reason") or "") != reason
                or canonical_json(new.to_dict()) != canonical_json(expected_new.to_dict())
                or (
                    content
                    and dict(new_snapshot.get("layer_hashes", {}) or {}).get(content_uri) != evidence_hash(content)
                )
            ):
                raise RedoIntegrityError("supersede SourceStore effect is incomplete")
            return
        if not operation.target_uri:
            raise RedoIntegrityError(f"regular redo {operation.action.value} has no target URI")
        target, snapshot = required(operation.target_uri)
        if operation.action == OperationAction.DELETE:
            if (
                target.lifecycle_state != LifecycleState.DELETED
                or target.metadata.get("delete_reason") != OperationAction.DELETE.value
            ):
                raise RedoIntegrityError("delete SourceStore effect is not the durable soft-delete state")
            return
        if operation.action == OperationAction.ARCHIVE:
            if (
                target.lifecycle_state != LifecycleState.ARCHIVED
                or target.metadata.get("archive_reason") != operation.payload.get("reason", "")
                or not target.metadata.get("archived_at")
            ):
                raise RedoIntegrityError("archive SourceStore effect is incomplete")
            return
        if operation.action == OperationAction.COMPRESS:
            layer_hashes = dict(snapshot.get("layer_hashes", {}) or {})
            if (
                target.lifecycle_state != LifecycleState.COLD
                or target.metadata.get("compression_reason") != operation.payload.get("reason", "")
                or not target.metadata.get("compressed_at")
                or not target.layers.l0_uri
                or not target.layers.l1_uri
                or not target.layers.l2_uri
                or any(layer_hashes.get(uri) is None for uri in target.layers.to_dict().values() if uri)
            ):
                raise RedoIntegrityError("compress SourceStore effect is incomplete")
            return
        if operation.action == OperationAction.REFRESH_LAYERS:
            layer_hashes = dict(snapshot.get("layer_hashes", {}) or {})
            if (
                not target.layers.l0_uri
                or not target.layers.l1_uri
                or not target.layers.l2_uri
                or any(layer_hashes.get(uri) is None for uri in target.layers.to_dict().values() if uri)
            ):
                raise RedoIntegrityError("layer refresh SourceStore effect is incomplete")
            return
        policy_actions = {
            OperationAction.REWARD,
            OperationAction.PENALIZE,
            OperationAction.COOLDOWN,
            OperationAction.SUPPRESS,
            OperationAction.DISABLE,
        }
        if operation.action in policy_actions and operation.context_type == ContextType.ACTION_POLICY:
            applied = {str(item) for item in target.metadata.get("applied_operation_ids", []) or []}
            if operation.operation_id not in applied:
                raise RedoIntegrityError("action-policy SourceStore effect is missing its operation id")
            return
        if operation.action == OperationAction.DISABLE:
            if (
                target.lifecycle_state != LifecycleState.DELETED
                or target.metadata.get("delete_reason") != OperationAction.DISABLE.value
            ):
                raise RedoIntegrityError("disable SourceStore effect is not durable")
            return
        if operation.action != OperationAction.REINDEX:
            raise RedoIntegrityError(f"unsupported regular redo action: {operation.action.value}")

    def _build_regular_relation_manifest(self, operation: ContextOperation) -> dict:
        """Bind the exact managed relation delta before the Source mutation."""

        expected: list[dict] = []
        previous: list[dict] = []
        if self.relation_store is not None:
            object_payload = operation.payload.get("context_object")
            desired = ContextObject.from_dict(object_payload) if isinstance(object_payload, dict) else None
            current: ContextObject | None = None
            current_uri = operation.target_uri or (desired.uri if desired is not None else None)
            if current_uri:
                try:
                    current = self.source_store.read_object(str(current_uri))
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                    current = None
            if operation.action in {OperationAction.ADD, OperationAction.UPDATE, OperationAction.MERGE}:
                if current is not None:
                    previous.extend(self._relation_specs_for_object(current))
                if desired is not None:
                    expected.extend(self._relation_specs_for_object(desired))
            elif operation.action == OperationAction.SUPERSEDE:
                if desired is not None:
                    try:
                        previous_new = self.source_store.read_object(desired.uri)
                    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                        previous_new = None
                    if previous_new is not None:
                        previous.extend(self._relation_specs_for_object(previous_new))
                    expected.extend(self._relation_specs_for_object(desired))
                if current is not None and desired is not None:
                    relation_metadata = {
                        "tenant_id": desired.tenant_id or current.tenant_id or "default",
                        "owner_user_id": desired.owner_user_id or current.owner_user_id,
                    }
                    expected.extend(
                        [
                            self._relation_spec(
                                desired.uri,
                                "supersedes",
                                current.uri,
                                relation_metadata,
                            ),
                            self._relation_spec(
                                current.uri,
                                "superseded_by",
                                desired.uri,
                                relation_metadata,
                            ),
                        ]
                    )
            elif current is not None:
                previous.extend(self._relation_specs_for_object(current))
                expected.extend(self._relation_specs_for_object(current))

        expected = self._unique_relation_specs(expected)
        previous = self._unique_relation_specs(previous)
        if any(self._regular_relation_has_canonical_endpoint(spec) for spec in expected):
            raise ValueError(
                "regular operations cannot publish relations to canonical memory; "
                "canonical relations require an immutable canonical receipt"
            )
        expected_keys = {self._relation_spec_key(spec) for spec in expected}
        remove = [
            self._relation_key_payload(spec) for spec in previous if self._relation_spec_key(spec) not in expected_keys
        ]
        remove = self._unique_relation_keys(remove)
        core = {
            "schema_version": "regular_relation_manifest_v1",
            "operation_id": operation.operation_id,
            "operation_fingerprint": self._operation_effect_fingerprint(operation),
            "user_id": operation.user_id,
            "tenant_id": self._regular_operation_tenant(operation),
            "context_type": operation.context_type.value,
            "target_uri": operation.target_uri,
            "expected": expected,
            "remove": remove,
        }
        return {**core, "fingerprint": stable_hash(core, length=64)}

    def _validate_regular_relation_manifest(
        self,
        operation: ContextOperation,
        manifest: dict | None,
    ) -> None:
        if not isinstance(manifest, dict):
            raise RedoIntegrityError("regular redo entry is missing its relation manifest")
        if manifest.get("schema_version") != "regular_relation_manifest_v1":
            raise RedoIntegrityError("regular redo relation manifest schema is unsupported")
        core = {key: value for key, value in manifest.items() if key != "fingerprint"}
        if manifest.get("fingerprint") != stable_hash(core, length=64):
            raise RedoIntegrityError("regular redo relation manifest fingerprint is corrupt")
        if (
            manifest.get("operation_id") != operation.operation_id
            or manifest.get("operation_fingerprint") != self._operation_effect_fingerprint(operation)
            or manifest.get("user_id") != operation.user_id
            or manifest.get("tenant_id") != self._regular_operation_tenant(operation)
            or manifest.get("context_type") != operation.context_type.value
            or manifest.get("target_uri") != operation.target_uri
            or not isinstance(manifest.get("expected"), list)
            or not isinstance(manifest.get("remove"), list)
        ):
            raise RedoIntegrityError("regular redo relation manifest crosses its operation boundary")
        expected = [dict(item) for item in manifest.get("expected", []) if isinstance(item, dict)]
        remove = [dict(item) for item in manifest.get("remove", []) if isinstance(item, dict)]
        if len(expected) != len(manifest.get("expected", [])) or len(remove) != len(manifest.get("remove", [])):
            raise RedoIntegrityError("regular redo relation manifest contains an invalid entry")
        if expected != self._unique_relation_specs(expected) or remove != self._unique_relation_keys(remove):
            raise RedoIntegrityError("regular redo relation manifest is not canonical")
        if any(self._regular_relation_has_canonical_endpoint(spec) for spec in expected):
            raise RedoIntegrityError("regular redo relation manifest crosses the canonical memory boundary")
        expected_keys = {self._relation_spec_key(spec) for spec in expected}
        if any(self._relation_spec_key(item) in expected_keys for item in remove):
            raise RedoIntegrityError("regular redo relation manifest removes an expected relation")

    def _apply_regular_relation_manifest(
        self,
        operation: ContextOperation,
        manifest: dict,
    ) -> None:
        self._validate_regular_relation_manifest(operation, manifest)
        if self.relation_store is None:
            if manifest.get("expected") or manifest.get("remove"):
                raise RedoIntegrityError("regular relation manifest requires a RelationStore")
            return
        for key in manifest.get("remove", []) or []:
            self.relation_store.delete_relation(
                str(key["source_uri"]),
                str(key["relation_type"]),
                str(key["target_uri"]),
            )
        self._ensure_relation_specs([dict(item) for item in manifest.get("expected", []) or []])
        self._validate_regular_relation_manifest_effect(manifest)

    def _validate_regular_relation_manifest_effect(self, manifest: dict) -> None:
        if self.relation_store is None:
            if manifest.get("expected") or manifest.get("remove"):
                raise RedoIntegrityError("regular relation effect has no RelationStore")
            return
        for spec in manifest.get("expected", []) or []:
            actual = {
                canonical_json(self._relation_effect_spec(relation))
                for relation in self.relation_store.relations_of(str(spec["source_uri"]))
            }
            if canonical_json(spec) not in actual:
                raise RedoIntegrityError("regular redo RelationStore effect is incomplete")
        for key in manifest.get("remove", []) or []:
            if any(
                relation.source_uri == key["source_uri"]
                and relation.relation_type == key["relation_type"]
                and relation.target_uri == key["target_uri"]
                for relation in self.relation_store.relations_of(str(key["source_uri"]))
            ):
                raise RedoIntegrityError("regular redo RelationStore retained a removed managed relation")

    def _relation_spec(
        self,
        source_uri: str,
        relation_type: str,
        target_uri: str,
        metadata: dict,
        *,
        weight: float = 1.0,
    ) -> dict:
        return {
            "source_uri": source_uri,
            "relation_type": relation_type,
            "target_uri": target_uri,
            "weight": float(weight),
            "metadata": {key: value for key, value in metadata.items() if value is not None},
        }

    def _regular_relation_has_canonical_endpoint(self, spec: dict) -> bool:
        for uri in (str(spec.get("source_uri") or ""), str(spec.get("target_uri") or "")):
            if not uri:
                continue
            if not uri.startswith("memoryos://"):
                continue
            if is_canonical_memory_uri(uri):
                return True
            try:
                obj = self.source_store.read_object(uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                continue
            if is_canonical_memory_object(obj):
                return True
        return False

    def _relation_spec_key(self, spec: dict) -> tuple[str, str, str]:
        return (
            str(spec.get("source_uri") or ""),
            str(spec.get("relation_type") or ""),
            str(spec.get("target_uri") or ""),
        )

    def _relation_key_payload(self, spec: dict) -> dict:
        source_uri, relation_type, target_uri = self._relation_spec_key(spec)
        return {
            "source_uri": source_uri,
            "relation_type": relation_type,
            "target_uri": target_uri,
        }

    def _unique_relation_specs(self, specs: list[dict]) -> list[dict]:
        unique = {canonical_json(spec): spec for spec in specs}
        return [unique[key] for key in sorted(unique)]

    def _unique_relation_keys(self, keys: list[dict]) -> list[dict]:
        unique = {canonical_json(key): key for key in keys}
        return [unique[key] for key in sorted(unique)]

    def _expected_regular_relation_specs(self, operation: ContextOperation) -> list[dict]:
        if self.relation_store is None:
            return []
        specs: list[dict] = []
        object_payload = operation.payload.get("context_object")
        if operation.action in {OperationAction.ADD, OperationAction.UPDATE, OperationAction.MERGE}:
            if isinstance(object_payload, dict):
                try:
                    obj = self.source_store.read_object(str(object_payload["uri"]))
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                    obj = ContextObject.from_dict(object_payload)
                specs.extend(self._relation_specs_for_object(obj))
        elif operation.action == OperationAction.SUPERSEDE:
            if operation.target_uri and isinstance(object_payload, dict):
                old_obj = self.source_store.read_object(operation.target_uri)
                try:
                    new_obj = self.source_store.read_object(str(object_payload["uri"]))
                except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                    new_obj = ContextObject.from_dict(object_payload)
                specs.extend(self._relation_specs_for_object(new_obj))
                metadata = {
                    "tenant_id": new_obj.tenant_id or old_obj.tenant_id or "default",
                    "owner_user_id": new_obj.owner_user_id or old_obj.owner_user_id,
                }
                specs.extend(
                    [
                        {
                            "source_uri": new_obj.uri,
                            "relation_type": "supersedes",
                            "target_uri": old_obj.uri,
                            "weight": 1.0,
                            "metadata": {key: value for key, value in metadata.items() if value is not None},
                        },
                        {
                            "source_uri": old_obj.uri,
                            "relation_type": "superseded_by",
                            "target_uri": new_obj.uri,
                            "weight": 1.0,
                            "metadata": {key: value for key, value in metadata.items() if value is not None},
                        },
                    ]
                )
        elif (
            operation.context_type == ContextType.ACTION_POLICY
            and operation.target_uri
            and operation.action
            in {
                OperationAction.REWARD,
                OperationAction.PENALIZE,
                OperationAction.COOLDOWN,
                OperationAction.SUPPRESS,
                OperationAction.DISABLE,
            }
        ):
            specs.extend(self._relation_specs_for_object(self.source_store.read_object(operation.target_uri)))
        unique = {canonical_json(spec): spec for spec in specs}
        return [unique[key] for key in sorted(unique)]

    def _restore_regular_relation_effect(self, operation: ContextOperation, source_effect: dict) -> None:
        expected = self._expected_regular_relation_specs(operation)
        if source_effect.get("relations") != expected:
            raise RedoIntegrityError("regular redo relation effect does not match its operation")
        self._ensure_relation_specs(expected)

    def _validate_regular_relation_postcondition(self, expected: list[dict]) -> None:
        if self.relation_store is None:
            if expected:
                raise RedoIntegrityError("regular redo relation effect has no RelationStore")
            return
        for spec in expected:
            actual = {
                canonical_json(self._relation_effect_spec(relation))
                for relation in self.relation_store.relations_of(str(spec["source_uri"]))
            }
            if canonical_json(spec) not in actual:
                raise RedoIntegrityError("regular redo RelationStore effect is incomplete")

    def _regular_source_effect_uris(self, operation: ContextOperation) -> list[str]:
        uris: list[str] = []
        if operation.target_uri:
            uris.append(str(operation.target_uri))
        object_payload = operation.payload.get("context_object")
        if isinstance(object_payload, dict) and object_payload.get("uri"):
            uris.append(str(object_payload["uri"]))
        return list(dict.fromkeys(uris))

    def _regular_operation_tenant(self, operation: ContextOperation) -> str:
        if not self._operation_matches_bound_tenant(operation):
            raise ValueError("regular operation tenant does not match bound tenant")
        return self.tenant_id

    def resume_canonical_batch(self, user_id: str, entries: list) -> list[str]:  # noqa: ANN001
        """从事务日志记录的阶段继续完成整批写入。"""

        operations = [entry.operation for entry in entries]
        if not operations:
            return []
        for entry in entries:
            self._validate_redo_boundary(
                user_id,
                entry.operation,
                source_effect=getattr(entry, "source_effect", None),
                relation_manifest=getattr(entry, "relation_manifest", None),
            )
            self._validate_canonical_artifact_keys(entry.operation)
        transaction_ids = {str(operation.payload.get("transaction_id", "")) for operation in operations}
        idempotency_keys = {str(operation.payload.get("idempotency_key", "")) for operation in operations}
        if len(transaction_ids) != 1 or "" in transaction_ids or len(idempotency_keys) != 1:
            raise ValueError("canonical recovery requires one complete transaction")
        transaction_id = next(iter(transaction_ids))
        idempotency_key = next(iter(idempotency_keys))
        outbox_path = self._outbox_path(transaction_id)
        self._reject_control_symlink(outbox_path, "canonical recovery outbox")
        try:
            prepared = validate_outbox(
                json.loads(outbox_path.read_text(encoding="utf-8")),
                transaction_id=transaction_id,
                idempotency_key=idempotency_key,
                tenant_id=self.tenant_id,
                user_id=user_id,
                operations=operations,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, OutboxIntegrityError) as exc:
            raise RedoIntegrityError("canonical recovery outbox envelope is invalid") from exc
        try:
            self.planning_proofs.load_canonical_intent(
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
                self.redo.commit(operation.operation_id)
            self.audit.record(
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
            self._validate_and_bind_operations(user_id, prepared_operations)
        except ValueError as exc:
            raise RedoIntegrityError("canonical recovery outbox crosses its user or tenant boundary") from exc
        for operation in prepared_operations:
            by_id.setdefault(operation.operation_id, operation)
        if set(expected_operation_ids) != set(by_id):
            raise RuntimeError("canonical recovery outbox is missing transaction operations")
        ordered = [by_id[operation_id] for operation_id in expected_operation_ids]
        head_was_published = any(entry.phase == "head_published" for entry in entries)
        marker = self._transaction_marker(idempotency_key)
        self._reject_control_symlink(marker, "canonical transaction receipt")
        if head_was_published:
            if not marker.exists():
                raise RedoIntegrityError(f"head-published redo transaction {transaction_id} has no immutable receipt")
            self._validate_head_published_receipt(
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
            self._validate_head_published_receipt(marker, receipt)
            diff = self._validate_transaction_marker(marker, ordered)
            for operation in ordered:
                self.redo.commit(operation.operation_id)
            return [operation.operation_id for operation in diff.operations]
        if prepared["status"] not in {"prepared", "source_committed"}:
            raise RedoIntegrityError("canonical recovery outbox is not recoverable")
        try:
            self._validate_canonical_envelope(user_id, ordered)
        except ValueError as exc:
            raise RedoIntegrityError("canonical recovery operations cross their user or tenant boundary") from exc
        self._preflight_canonical_revisions(ordered, check_revisions=False)
        self._validate_authoritative_batch(ordered)
        if not marker.exists():
            self.final_state_validator.validate(
                ordered,
                tenant_id=self.tenant_id,
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
            self._validate_canonical_relation_manifest(
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
                self._validate_canonical_source_effect(
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
                if self._canonical_pending_effect(operation) and operation.target_uri
            ),
        }
        with ExitStack() as locks:
            guards: list[LeaseGuard] = []
            for lock_key in sorted(lock_keys):
                guards.append(locks.enter_context(self.path_lock.acquire(self._lock_key(lock_key))))
            with self.path_lock.fenced(guards):
                if marker.exists():
                    receipt = load_transaction_receipt(marker)
                    if head_was_published:
                        self._validate_head_published_receipt(marker, receipt)
                    else:
                        publish_current_head_sets(self.artifact_root, marker, receipt)
                        self._mark_current_heads_published(ordered)
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
                with self.path_lock.fenced(guards):
                    payload = operation.payload.get("context_object")
                    if not isinstance(payload, dict):
                        raise ValueError("canonical recovery requires context_object")
                    uri = str(payload["uri"])
                    if self._canonical_pending_effect(operation):
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
                        self._apply_canonical_relation_manifest(
                            operation,
                            relation_manifests[operation.operation_id],
                        )
                        source_effect = self._capture_canonical_source_effect(
                            operation,
                            relation_manifests[operation.operation_id],
                        )
                        self.redo.advance(
                            operation,
                            phase="source_written",
                            source_effect=source_effect,
                            relation_manifest=relation_manifests[operation.operation_id],
                        )
                        self.audit.record(
                            user_id,
                            "canonical_memory_operation_applied_during_recovery",
                            operation.to_dict(),
                        )
                        self.redo.advance(operation, phase="audit_written")
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
                    self._apply_canonical_relation_manifest(
                        operation,
                        relation_manifests[operation.operation_id],
                    )
                    source_effect = self._capture_canonical_source_effect(
                        operation,
                        relation_manifests[operation.operation_id],
                    )
                    self.redo.advance(
                        operation,
                        phase="source_written",
                        source_effect=source_effect,
                        relation_manifest=relation_manifests[operation.operation_id],
                    )
                    self.audit.record(
                        user_id, "canonical_memory_operation_applied_during_recovery", operation.to_dict()
                    )
                    self.redo.advance(operation, phase="audit_written")
                    operation.status = OperationStatus.COMMITTED
            with self.path_lock.fenced(guards):
                self._write_outbox_event(
                    transaction_id,
                    idempotency_key,
                    ordered,
                    status="source_committed",
                    relation_manifests=relation_manifests,
                )
                diff = self._ensure_canonical_transaction_diff(
                    user_id,
                    transaction_id,
                    ordered,
                )
                self._write_transaction_marker(
                    marker,
                    diff,
                    ordered,
                    relation_manifests=relation_manifests,
                )
                publish_current_head_sets(
                    self.artifact_root,
                    marker,
                    load_transaction_receipt(marker),
                )
                self._mark_current_heads_published(ordered)
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

    def recover_pending_canonical(
        self,
        user_id: str,
        *,
        commit_group_id: str | None = None,
    ) -> list[str]:
        """恢复卡在准备阶段或源数据已写入阶段的记忆事务。"""

        grouped: dict[str, list] = {}
        for entry in self.redo.pending_entries():
            if (
                entry.operation.user_id != user_id
                or not self._operation_matches_bound_tenant(entry.operation)
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
            recovered.extend(self.resume_canonical_batch(user_id, entries))
        return recovered

    def recover_pending_regular_memory(
        self,
        user_id: str,
        *,
        commit_group_id: str,
    ) -> list[str]:
        """Finish redo-backed pending-memory effects for one session commit group."""

        recovered: list[str] = []
        for entry in self.redo.pending_entries():
            operation = entry.operation
            if (
                operation.user_id != user_id
                or not self._operation_matches_bound_tenant(operation)
                or operation.payload.get("canonical_memory") is True
                or operation.payload.get("canonical_pending_proposal") is not True
                or operation.payload.get("commit_consumer")
                or str(operation.payload.get("commit_group_id") or "") != commit_group_id
            ):
                continue
            if self.resume(
                user_id,
                operation,
                entry.phase,
                source_effect=entry.source_effect,
                relation_manifest=entry.relation_manifest,
            ):
                recovered.append(operation.operation_id)
        return recovered

    def committed_canonical_diffs(
        self,
        user_id: str,
        commit_group_id: str,
    ) -> list[ContextDiff]:
        """Load every integrity-checked transaction marker bound to one commit group."""

        root = self.artifact_root / "system" / "transactions"
        if not root.exists():
            return []
        result: list[ContextDiff] = []
        for path in sorted(root.glob("*.json")):
            if path.is_symlink():
                raise ValueError("canonical transaction receipt cannot be a symbolic link")
            diff = self._transaction_marker_diff(path)
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
            diff = self._validate_transaction_marker(path, diff.operations)
            result.append(diff)
        return result

    def committed_memory_effect_diffs(
        self,
        user_id: str,
        commit_group_id: str,
    ) -> list[ContextDiff]:
        """Load marker-backed canonical and pending-memory effects for one group."""

        result = self.committed_canonical_diffs(user_id, commit_group_id)
        root = self.artifact_root / "system" / "operations"
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
            self._validate_and_bind_operations(user_id, [operation])
            stored = self._validate_operation_marker(path, operation)
            result.append(
                ContextDiff(
                    user_id=user_id,
                    operations=[stored],
                    diff_id=f"diff_{stored.operation_id}",
                    created_at=stored.created_at,
                )
            )
        return result

    def _write_recovery_diff(self, user_id: str, operation: ContextOperation) -> None:
        self._ensure_single_operation_diff(user_id, operation)

    def _operation_marker(self, operation_id: str) -> Path:
        key = require_safe_path_segment(operation_id, "operation_id")
        return self.artifact_root / "system" / "operations" / f"{key}.json"

    @staticmethod
    def _regular_lock_keys(operation: ContextOperation) -> tuple[str, ...]:
        """Fence both mutable target state and the immutable operation identity."""

        target = operation.target_uri or f"{operation.user_id}:{operation.operation_id}"
        return tuple(sorted({target, f"operation-id:{operation.operation_id}"}))

    def _write_operation_marker(
        self,
        operation: ContextOperation,
        *,
        source_effect: dict | None,
        relation_manifest: dict | None,
        diff: ContextDiff,
    ) -> None:
        if operation.payload.get("canonical_memory") is True:
            return
        self._validate_regular_recovery_effect(
            operation.user_id,
            operation,
            source_effect,
            relation_manifest=relation_manifest,
        )
        path = self._operation_marker(operation.operation_id)
        self._reject_control_symlink(path, "operation receipt")
        if operation.payload.get("canonical_pending_proposal") is True:
            self._bind_pending_receipt_identity(operation)
            if path.exists():
                stored = self._validate_operation_marker(path, operation)
                self._publish_pending_current_head(path, stored)
                self._mark_current_heads_published([operation])
                return
            planning_digest = self._ensure_pending_planning_digest(operation)
            try:
                intent = self.planning_proofs.load_pending_intent(
                    operation.operation_id,
                    operation=operation,
                    relation_manifest=relation_manifest,
                )
            except PlanningProofIntegrityError as exc:
                raise ValueError("pending receipt requires its pre-write prepared intent") from exc
            intent_digest = str(intent["prepared_intent_digest"])
            receipt = build_transaction_receipt(
                transaction_id=operation.operation_id,
                idempotency_key=str(operation.payload.get("idempotency_key") or operation.operation_id),
                tenant_id=self.tenant_id,
                user_id=operation.user_id,
                commit_group_id=str(operation.payload.get("commit_group_id") or ""),
                operations=[operation],
                diff=diff.to_dict(),
                planning_digest=planning_digest,
                prepared_intent_digest=intent_digest,
                prepared_intent_schema_version=PENDING_PREPARED_INTENT_SCHEMA_VERSION,
                relation_effects=relation_effects_from_manifest(relation_manifest),
                created_at=diff.created_at,
            )
            self._notify("before_receipt", operation.operation_id)
            self._reject_control_symlink(path, "pending operation receipt")
            atomic_create_json(path, receipt, artifact_root=self.artifact_root)
            self._notify("after_receipt", operation.operation_id)
            self._notify("before_current_head", operation.operation_id)
            publish_current_head_sets(self.artifact_root, path, receipt)
            self._mark_current_heads_published([operation])
            self._notify("after_current_head", operation.operation_id)
            return
        stored_operation = operation.to_dict()
        stored_operation["status"] = OperationStatus.COMMITTED.value
        if path.exists():
            self._validate_operation_marker(path, operation)
            return
        object_effects = []
        for uri in self._regular_source_effect_uris(operation):
            logical_absence = operation.action == OperationAction.DELETE and uri == operation.target_uri
            object_effects.append(
                object_effect_from_store(
                    self.source_store,
                    uri,
                    operation_type=operation.action.value,
                    expected_exists=not logical_absence,
                    logical_absence=logical_absence,
                )
            )
        payload = build_marker(
            transaction_id=operation.operation_id,
            idempotency_key=operation.operation_id,
            tenant_id=self.tenant_id,
            user_id=operation.user_id,
            operation_ids=[operation.operation_id],
            object_effects=object_effects,
            relation_effects=relation_effects_from_manifest(relation_manifest),
            diff=diff.to_dict(),
            operations=[stored_operation],
        )
        payload.update(
            {
                "operation_id": operation.operation_id,
                "action": operation.action.value,
                "context_type": operation.context_type.value,
                "target_uri": operation.target_uri,
                "commit_group_id": operation.payload.get("commit_group_id"),
                "commit_consumer": operation.payload.get("commit_consumer"),
                "effect_fingerprint": self._operation_effect_fingerprint(operation),
                "operation": stored_operation,
            }
        )
        core = {key: value for key, value in payload.items() if key != "marker_digest"}
        payload["marker_digest"] = canonical_digest(core)
        self._reject_control_symlink(path, "operation marker")
        atomic_create_json(path, payload, artifact_root=self.artifact_root)

    def _bind_pending_receipt_identity(self, operation: ContextOperation) -> None:
        """Bind a pending lifecycle operation before Source/diff/receipt publication."""

        commit_group_id = operation.payload.get("commit_group_id")
        if not isinstance(commit_group_id, str) or not commit_group_id:
            raise ValueError("pending lifecycle operation requires a commit group identity")
        operation.payload.update(
            {
                "transaction_id": operation.operation_id,
                "idempotency_key": str(operation.payload.get("idempotency_key") or operation.operation_id),
                "tenant_id": self.tenant_id,
            }
        )

    def _publish_pending_current_head(
        self,
        path: Path,
        operation: ContextOperation,
    ) -> None:
        if operation.payload.get("canonical_pending_proposal") is not True:
            return
        try:
            receipt = load_transaction_receipt(path)
        except ReceiptIntegrityError as exc:
            raise ValueError("pending operation receipt is corrupt") from exc
        publish_current_head_sets(self.artifact_root, path, receipt)

    def _validate_operation_marker(self, path: Path, operation: ContextOperation) -> ContextOperation:
        self._validate_and_bind_operations(operation.user_id, [operation])
        self._reject_control_symlink(path, "operation receipt")
        try:
            raw_payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("operation marker is unreadable") from exc
        if raw_payload.get("schema_version") == TRANSACTION_RECEIPT_SCHEMA_VERSION:
            try:
                receipt = validate_transaction_receipt(
                    raw_payload,
                    transaction_id=operation.operation_id,
                    tenant_id=self.tenant_id,
                    user_id=operation.user_id,
                    operation_ids=[operation.operation_id],
                )
            except ReceiptIntegrityError as exc:
                raise ValueError("pending operation receipt is corrupt") from exc
            stored_payloads = receipt.get("operations", [])
            if len(stored_payloads) != 1 or not isinstance(stored_payloads[0], dict):
                raise ValueError("pending operation receipt has invalid membership")
            stored = ContextOperation.from_dict(stored_payloads[0])
            self._validate_and_bind_operations(operation.user_id, [stored])
            requested = operation
            if requested.target_uri is None and stored.target_uri is not None:
                requested = ContextOperation.from_dict(operation.to_dict())
                requested.target_uri = stored.target_uri
            if self._operation_effect_fingerprint(stored) != self._operation_effect_fingerprint(requested):
                raise ValueError("operation idempotency receipt conflicts with the requested effect")
            if stored.payload.get("canonical_pending_proposal") is True:
                self._ensure_pending_planning_digest(stored)
                try:
                    self.planning_proofs.load_pending_intent(
                        stored.operation_id,
                        operation=stored,
                        prepared_intent_digest=str(receipt.get("prepared_intent_digest") or ""),
                    )
                except PlanningProofIntegrityError as exc:
                    raise ValueError("pending operation receipt is detached from its prepared intent") from exc
            stored.status = OperationStatus.COMMITTED
            return stored
        try:
            payload = validate_marker(
                path,
                self.source_store,
                self.relation_store,
                transaction_id=operation.operation_id,
                idempotency_key=operation.operation_id,
                tenant_id=self.tenant_id,
                user_id=operation.user_id,
                operation_ids=[operation.operation_id],
            )
        except EffectProofError as exc:
            if path.exists():
                quarantine_control_file(
                    self.artifact_root,
                    path,
                    kind="operation_marker",
                    error=exc,
                    identifiers={"operation_id": operation.operation_id},
                )
            raise ValueError("operation marker cannot prove its durable effect") from exc
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
        self._validate_and_bind_operations(operation.user_id, [stored])
        if operation.target_uri not in {None, stored.target_uri} or payload.get("target_uri") != stored.target_uri:
            raise ValueError("operation idempotency marker conflicts with the requested target")
        requested = operation
        if requested.target_uri is None and stored.target_uri is not None:
            requested = ContextOperation.from_dict(operation.to_dict())
            requested.target_uri = stored.target_uri
        if payload.get("effect_fingerprint") != self._operation_effect_fingerprint(stored) or payload.get(
            "effect_fingerprint"
        ) != self._operation_effect_fingerprint(requested):
            raise ValueError("operation idempotency marker conflicts with the requested effect")
        stored.status = OperationStatus.COMMITTED
        return stored

    def _refresh_regular_effect_proofs(self, changed_uris: list[str]) -> None:
        """Atomically advance prior regular markers to the current Source fact."""

        wanted = set(changed_uris)
        marker_root = self.artifact_root / "system" / "operations"
        if not wanted or not marker_root.exists():
            return
        for path in sorted(marker_root.glob("*.json")):
            try:
                if path.is_symlink():
                    raise OSError("operation marker cannot be a symbolic link")
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                if path.exists():
                    quarantine_control_file(
                        self.artifact_root,
                        path,
                        kind="operation_marker",
                        error=exc,
                        identifiers={"operation_id": path.stem},
                    )
                raise RedoIntegrityError("regular operation marker is unreadable") from exc
            if isinstance(payload, dict) and payload.get("schema_version") == TRANSACTION_RECEIPT_SCHEMA_VERSION:
                # Immutable pending receipts are historical facts; a later
                # lifecycle revision must never refresh their effect proof.
                continue
            if not isinstance(payload, dict) or payload.get("schema_version") != "effect_marker_v1":
                quarantine_control_file(
                    self.artifact_root,
                    path,
                    kind="operation_marker",
                    error=ValueError("unsupported marker schema"),
                    identifiers={"operation_id": path.stem},
                )
                raise RedoIntegrityError("regular operation marker schema is unsupported")
            digest = payload.get("marker_digest")
            core = {key: value for key, value in payload.items() if key != "marker_digest"}
            if not isinstance(digest, str) or digest != canonical_digest(core):
                quarantine_control_file(
                    self.artifact_root,
                    path,
                    kind="operation_marker",
                    error=ValueError("marker digest mismatch"),
                    identifiers={"operation_id": path.stem},
                )
                raise RedoIntegrityError("regular operation marker digest is corrupt")
            effects = payload.get("object_effects")
            if not isinstance(effects, list) or not any(
                isinstance(effect, dict) and str(effect.get("uri") or "") in wanted for effect in effects
            ):
                continue
            refreshed: list[dict] = []
            for effect in effects:
                if (
                    not isinstance(effect, dict)
                    or str(effect.get("uri") or "") not in wanted
                    or effect.get("expected_exists") is not True
                ):
                    refreshed.append(effect)
                    continue
                refreshed.append(
                    object_effect_from_store(
                        self.source_store,
                        str(effect["uri"]),
                        operation_type=str(effect.get("operation_type") or "UPDATE"),
                    )
                )
            payload["object_effects"] = refreshed
            relation_effects = payload.get("relation_effects")
            if self.relation_store is not None and isinstance(relation_effects, list):
                refreshed_relations: list[dict] = []
                for effect in relation_effects:
                    if not isinstance(effect, dict) or effect.get("expected_exists") is not True:
                        refreshed_relations.append(effect)
                        continue
                    identity = relation_identity(dict(effect.get("identity", {}) or {}))
                    if not ({identity["source_uri"], identity["target_uri"]} & wanted):
                        refreshed_relations.append(effect)
                        continue
                    matches = [
                        relation
                        for relation in self.relation_store.relations_of(identity["source_uri"])
                        if relation.source_uri == identity["source_uri"]
                        and relation.relation_type == identity["relation_type"]
                        and relation.target_uri == identity["target_uri"]
                    ]
                    if len(matches) != 1:
                        refreshed_relations.append(effect)
                        continue
                    normalized = normalized_relation(matches[0])
                    refreshed_relations.append(
                        {
                            **effect,
                            "identity": identity,
                            **identity,
                            "relation": normalized,
                            "relation_digest": canonical_digest(normalized),
                        }
                    )
                payload["relation_effects"] = refreshed_relations
            updated_core = {key: value for key, value in payload.items() if key != "marker_digest"}
            payload["marker_digest"] = canonical_digest(updated_core)
            atomic_write_json(path, payload, artifact_root=self.artifact_root)

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
                canonical_kind = str(dict(obj.metadata or {}).get("canonical_kind") or "")
                if content and canonical_kind not in {"slot", "claim", "pending_proposal"}:
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
        del operation
        if self.relation_store is None:
            return
        self._ensure_relation_specs(self._relation_specs_for_object(obj))

    def _relation_specs_for_object(self, obj: ContextObject) -> list[dict]:
        metadata = dict(obj.metadata)
        relation_metadata = {"tenant_id": obj.tenant_id or "default", "owner_user_id": obj.owner_user_id}
        specs: list[dict] = []

        def add(source_uri: str, relation_type: str, target_uri: str, relation_meta: dict) -> None:
            if not target_uri:
                return
            specs.append(
                {
                    "source_uri": source_uri,
                    "relation_type": relation_type,
                    "target_uri": target_uri,
                    "weight": 1.0,
                    "metadata": {key: value for key, value in relation_meta.items() if value is not None},
                }
            )

        if obj.context_type == ContextType.ACTION_POLICY:
            add(obj.uri, "anchored_by", str(metadata.get("memory_anchor_uri", "")), relation_metadata)
            for uri in metadata.get("required_resource_uris", []) or []:
                add(obj.uri, "requires_resource", str(uri), relation_metadata)
            for uri in metadata.get("required_skill_uris", []) or []:
                add(obj.uri, "requires_skill", str(uri), relation_metadata)
            for uri in metadata.get("supported_behavior_pattern_uris", []) or []:
                add(obj.uri, "supported_by", str(uri), relation_metadata)
            for uri in metadata.get("constrained_by_memory_uris", []) or []:
                add(obj.uri, "constrained_by", str(uri), relation_metadata)
        elif obj.context_type in {ContextType.BEHAVIOR_PATTERN, ContextType.BEHAVIOR_CLUSTER}:
            add(obj.uri, "anchored_by", str(metadata.get("memory_anchor_uri", "")), relation_metadata)
            for uri in metadata.get("case_refs", []) or []:
                add(obj.uri, "aggregated_from", str(uri), relation_metadata)
            for uri in metadata.get("related_policy_uris", []) or metadata.get("policy_uris", []) or []:
                add(str(uri), "supported_by", obj.uri, relation_metadata)
        elif obj.context_type == ContextType.MEMORY:
            for policy_uri in metadata.get("constrains_policy_uris", []) or []:
                add(str(policy_uri), "constrained_by", obj.uri, relation_metadata)
            for behavior_uri in metadata.get("supporting_behavior_uris", []) or []:
                add(obj.uri, "evidence_for", str(behavior_uri), relation_metadata)
        for relation in obj.relations:
            specs.append(self._relation_effect_spec(relation))
        unique = {canonical_json(spec): spec for spec in specs}
        return [unique[key] for key in sorted(unique)]

    def _add_relation(self, source_uri: str, relation_type: str, target_uri: str, metadata: dict) -> None:
        if self.relation_store is None or not target_uri:
            return
        self._ensure_relation_specs(
            [
                {
                    "source_uri": source_uri,
                    "relation_type": relation_type,
                    "target_uri": target_uri,
                    "weight": 1.0,
                    "metadata": {key: value for key, value in metadata.items() if value is not None},
                }
            ]
        )

    def _ensure_relation_specs(self, specs: list[dict]) -> None:
        if self.relation_store is None:
            return
        for spec in specs:
            existing = self.relation_store.relations_of(str(spec["source_uri"]))
            matching_key = next(
                (
                    relation
                    for relation in existing
                    if relation.source_uri == spec["source_uri"]
                    and relation.relation_type == spec["relation_type"]
                    and relation.target_uri == spec["target_uri"]
                ),
                None,
            )
            if matching_key is not None and self._relation_effect_spec(matching_key) == spec:
                continue
            if matching_key is not None:
                self.relation_store.delete_relation(
                    matching_key.source_uri,
                    matching_key.relation_type,
                    matching_key.target_uri,
                )
            self.relation_store.add_relation(
                ContextRelation(
                    source_uri=str(spec["source_uri"]),
                    relation_type=str(spec["relation_type"]),
                    target_uri=str(spec["target_uri"]),
                    weight=float(spec.get("weight", 1.0)),
                    metadata=dict(spec.get("metadata", {}) or {}),
                )
            )

    def _relation_effect_spec(self, relation: ContextRelation) -> dict:
        return {
            "source_uri": relation.source_uri,
            "relation_type": relation.relation_type,
            "target_uri": relation.target_uri,
            "weight": float(relation.weight),
            "metadata": dict(relation.metadata),
        }

    def _read_content_or_empty(self, uri: str) -> str:
        try:
            return self.source_store.read_content(uri)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return ""
