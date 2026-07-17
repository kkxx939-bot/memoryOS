"""Implementation component for CommitStateMachine.

The public OperationCommitter delegates explicitly to this component so fault
injection hooks remain available on the facade.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.core.ids import require_safe_path_segment, stable_hash
from memoryos.core.integrity import canonical_json
from memoryos.operations.commit.redo_log import RedoEntry, RedoIntegrityError
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class CommitStateMachine:
    """Own the CommitStateMachine responsibility of a commit."""

    @staticmethod
    def _delete_tombstone_ids(operation: ContextOperation) -> tuple[str, ...]:
        raw = operation.payload.get("projection_tombstone_ids", ())
        if not isinstance(raw, list | tuple) or any(not isinstance(item, str) or not item for item in raw):
            return ()
        return tuple(dict.fromkeys(raw))

    @staticmethod
    def _require_delete_tombstone_capability(committer, operations: list[ContextOperation]) -> None:
        """Reject a production DELETE before its first durable or Source effect.

        InMemoryIndexStore remains a deliberately small compatibility test
        backend.  A Catalog store advertising the durable tombstone API may
        never fall back to synchronous ``delete_index``.
        """

        if not any(operation.action == OperationAction.DELETE for operation in operations):
            return
        durable_catalog = callable(getattr(committer.index_store, "enqueue_tombstone", None))
        if durable_catalog and committer.tombstone_service is None:
            raise RuntimeError("production DELETE requires ProjectionTombstoneService")

    @staticmethod
    def _prepare_delete_tombstones(
        committer,
        operation: ContextOperation,
        *,
        trust_durable_binding: bool = False,
    ) -> tuple[str, ...]:
        """Journal every derived projection before retiring the Source object."""

        if operation.action != OperationAction.DELETE or not operation.target_uri:
            return ()
        bound = committer._delete_tombstone_ids(operation)
        if trust_durable_binding and bound:
            return bound
        if committer.tombstone_service is None:
            if callable(getattr(committer.index_store, "enqueue_tombstone", None)):
                raise RuntimeError("production DELETE requires ProjectionTombstoneService")
            operation.payload.pop("projection_tombstone_ids", None)
            return ()
        enqueue = getattr(committer.tombstone_service, "enqueue_uri", None)
        if not callable(enqueue):
            raise TypeError("ProjectionTombstoneService has no durable URI enqueue operation")
        raw_ids = enqueue(
            operation.target_uri,
            tenant_id=committer.tenant_id,
            reason=str(operation.payload.get("reason") or OperationAction.DELETE.value),
            require_source_retired=True,
        )
        if not isinstance(raw_ids, list | tuple):
            raise RuntimeError("durable projection tombstone journal returned an invalid result")
        tombstone_ids = tuple(dict.fromkeys(str(item) for item in raw_ids if str(item)))
        if not tombstone_ids:
            raise RuntimeError("production DELETE did not journal a projection tombstone")
        operation.payload["projection_tombstone_ids"] = list(tombstone_ids)
        return tombstone_ids

    @staticmethod
    def _settle_delete_tombstones(committer, operations: list[ContextOperation]) -> None:
        """Replay each committed DELETE's exact durable projection journal."""

        if committer.tombstone_service is None:
            return
        process = getattr(committer.tombstone_service, "process_tombstones", None)
        if not callable(process):
            raise TypeError("ProjectionTombstoneService has no exact replay operation")
        for operation in operations:
            if operation.action != OperationAction.DELETE:
                continue
            tombstone_ids = committer._delete_tombstone_ids(operation)
            if not tombstone_ids:
                raise RuntimeError("committed production DELETE has no durable tombstone binding")
            result = process(tombstone_ids)
            failed = getattr(result, "failed", None)
            processed = getattr(result, "processed", None)
            stale = getattr(result, "stale", None)
            if failed is None or processed is None or stale is None:
                raise RuntimeError("durable projection tombstone cleanup returned an invalid result")
            if failed:
                raise RuntimeError("derived projection tombstone cleanup is retryable but incomplete")

    @staticmethod
    @contextmanager
    def _durable_startup_recovery_scope(committer, group_id: str) -> Iterator[None]:
        """Authorize commits only for one already-durable startup group.

        The runtime builder is the sole production caller.  The final
        committer still reloads and validates the group, archive, planning
        envelope and operation bindings for every commit made in this scope.
        """

        require_safe_path_segment(group_id, "startup recovery commit_group_id")
        readiness = getattr(committer.source_store, "readiness", None)
        state = getattr(getattr(readiness, "state", None), "value", "")
        if state != "RECOVERING":
            raise RuntimeError("durable startup commit scope requires a RECOVERING runtime")
        token = committer._startup_recovery_group.set(group_id)
        try:
            yield
        finally:
            committer._startup_recovery_group.reset(token)

    @staticmethod
    @contextmanager
    def _migration_projection_fence(committer) -> Iterator[None]:
        """Serialize all Source mutations with a tenant serving rebuild.

        ``commit`` recursively handles mixed canonical/regular batches and
        recovery entry points call one another.  Context-local depth keeps
        those nested calls reentrant without reacquiring the non-reentrant
        durable SQLite lease.
        """

        depth = committer._projection_fence_depth.get()
        if depth:
            depth_token = committer._projection_fence_depth.set(depth + 1)
            try:
                yield
            finally:
                committer._projection_fence_depth.reset(depth_token)
            return
        acquire = getattr(committer.migration_gate, "acquire_projection_fence", None)
        release = getattr(committer.migration_gate, "release_projection_fence", None)
        fence = acquire() if callable(acquire) else None
        depth_token = committer._projection_fence_depth.set(1)
        try:
            yield
        finally:
            committer._projection_fence_depth.reset(depth_token)
            if callable(release):
                release(fence)

    @staticmethod
    def _require_commit_ready(
        committer,
        user_id: str,
        operations: list[ContextOperation],
    ) -> None:
        readiness = getattr(committer.source_store, "readiness", None)
        require_ready = getattr(readiness, "require_ready", None)
        if not callable(require_ready):
            return
        state = str(getattr(getattr(readiness, "state", None), "value", ""))
        if state == "READY":
            return
        group_id = committer._startup_recovery_group.get()
        if state == "RECOVERING" and group_id:
            committer._validate_durable_startup_commit(group_id, user_id, operations)
            return
        require_ready()

    @staticmethod
    def _validate_durable_startup_commit(
        committer,
        group_id: str,
        user_id: str,
        operations: list[ContextOperation],
    ) -> None:
        """Independently bind a RECOVERING commit to durable semantic input."""

        from memoryos.operations.commit.commit_group import CommitGroupStore

        group = CommitGroupStore(committer.artifact_root).load(group_id)
        if (
            group is None
            or group.group_id != group_id
            or group.user_id != user_id
            or group.tenant_id != committer.tenant_id
            or group.complete
        ):
            raise RuntimeError("startup commit is detached from its durable commit group")
        archive = committer._session_evidence_reader().read_archive(
            group.archive_uri,
            tenant_id=committer.tenant_id,
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
            envelope = cast(dict, committer.planning_envelopes.load_validated_payload(group.task_id))
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

    @staticmethod
    def _notify(committer, stage: str, transaction_id: str) -> None:
        if callable(committer.test_hook):
            committer.test_hook(stage, transaction_id)

    @staticmethod
    def _mark_current_heads_published(
        committer,
        operations: list[ContextOperation],
    ) -> None:
        """Persist the post-head crash boundary before any post-head hook."""

        for operation in operations:
            committer.redo.advance(operation, phase="head_published")

    @staticmethod
    def _validate_head_published_receipt(
        committer,
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
                head, bound_receipt, _snapshot = committer._load_canonical_current_head(uri)
            except (FileNotFoundError, committer._canonical_current_head_error()) as exc:
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
            elif str(head.get("receipt_digest") or "") != str(receipt.get("receipt_digest") or "") or str(
                bound_receipt.get("receipt_digest") or ""
            ) != str(receipt.get("receipt_digest") or ""):
                raise RedoIntegrityError(
                    "head-published redo transaction "
                    f"{transaction_id} current head for {uri} is detached from its receipt"
                )
            try:
                # A current head is a proof of the complete live bundle, not
                # merely a pointer to an immutable receipt.  This committed
                # read also preserves an older snapshot when a separately
                # proved pre-head transaction is legitimately in flight.
                committer._read_committed_canonical(uri)
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                raise RedoIntegrityError(
                    f"head-published redo transaction {transaction_id} current Source bundle for {uri} is invalid"
                ) from exc
        if not receipt_path.exists():
            raise RedoIntegrityError(
                f"head-published redo transaction {transaction_id} is missing its immutable receipt"
            )

    @staticmethod
    def _lock_key(committer, raw_key: str) -> str:
        # The default tenant keeps its historical lock key. Non-default
        # tenants have physically distinct artifacts and therefore receive a
        # tenant-qualified key in the shared lock store.
        canonical_key = raw_key
        if raw_key.startswith("memoryos://"):
            canonical_key = str(ContextURI.parse(raw_key))
        return canonical_key if committer.tenant_id == "default" else f"tenant:{committer.tenant_id}:{canonical_key}"

    @staticmethod
    def _validate_tenant_id(value: object, label: str) -> str:
        if not isinstance(value, str) or not value.strip() or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"{label} must be one safe non-empty path segment")
        return value

    @staticmethod
    def _explicit_tenant_declarations(committer, operation: ContextOperation) -> list[tuple[str, str]]:
        payload = operation.payload
        if not isinstance(payload, dict):
            raise ValueError("operation payload must be an object")
        declarations: list[tuple[str, str]] = []

        def inspect(container: object, path: str) -> None:
            if not isinstance(container, dict) or "tenant_id" not in container:
                return
            declarations.append((path, committer._validate_tenant_id(container["tenant_id"], f"{path}.tenant_id")))

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

    @staticmethod
    def _operation_matches_bound_tenant(committer, operation: ContextOperation) -> bool:
        try:
            declarations = committer._explicit_tenant_declarations(operation)
        except ValueError:
            return False
        return all(value == committer.tenant_id for _, value in declarations)

    @staticmethod
    def _validate_and_bind_operations(
        committer,
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
            declarations = committer._explicit_tenant_declarations(operation)
            if any(value != committer.tenant_id for _, value in declarations):
                paths = ", ".join(path for path, value in declarations if value != committer.tenant_id)
                raise ValueError(f"operation tenant does not match bound tenant: {paths}")
            declarations_by_operation.append((operation, declarations))

        # Bind only after every operation has passed so a rejected batch is not
        # partially normalized and no artifact is written with an implicit tenant.
        for operation, _ in declarations_by_operation:
            operation.payload.setdefault("tenant_id", committer.tenant_id)
            object_payload = operation.payload.get("context_object")
            if isinstance(object_payload, dict):
                object_payload.setdefault("tenant_id", committer.tenant_id)

    @staticmethod
    def _validate_recovery_artifact_tenant(committer, payload: object, label: str) -> None:
        if not isinstance(payload, dict) or "tenant_id" not in payload:
            return
        tenant = committer._validate_tenant_id(payload["tenant_id"], f"{label} tenant_id")
        if tenant != committer.tenant_id:
            raise RedoIntegrityError(f"{label} crosses the bound tenant")

    @staticmethod
    def _validate_redo_boundary(
        committer,
        user_id: str,
        operation: ContextOperation,
        *,
        source_effect: dict | None = None,
        relation_manifest: dict | None = None,
    ) -> None:
        try:
            committer._validate_and_bind_operations(user_id, [operation])
        except ValueError as exc:
            raise RedoIntegrityError("redo operation crosses its user or tenant boundary") from exc
        try:
            # Recovery is an alternate write entry point, so it must enforce
            # the same canonical/pending classification as a fresh commit.
            # Otherwise a legacy or hand-built regular redo could rewrite a
            # receipt-backed object before regular postcondition validation
            # notices the incompatible materialization.
            committer._reject_canonical_regular_bypass([operation])
        except ValueError as exc:
            raise RedoIntegrityError(
                "canonical memory redo recovery cannot bypass its committed transaction boundary"
            ) from exc
        committer._validate_recovery_artifact_tenant(source_effect, "redo source effect")
        committer._validate_recovery_artifact_tenant(relation_manifest, "redo relation manifest")

    @staticmethod
    def _load_exact_redo_entry(
        committer,
        user_id: str,
        operation: ContextOperation,
        phase: str,
        *,
        source_effect: dict | None,
        relation_manifest: dict | None,
    ) -> RedoEntry:
        """Bind recovery to the single integrity-checked durable redo entry.

        ``resume`` is a recovery entry point, not a second write API.  Caller
        supplied operation data therefore cannot select another tombstone,
        Source effect, relation manifest, or recovery phase.
        """

        matches = [entry for entry in committer.redo.pending_entries() if entry.operation_id == operation.operation_id]
        if len(matches) != 1:
            raise RedoIntegrityError("redo recovery requires exactly one durable operation entry")
        entry = matches[0]
        if entry.user_id != user_id or not committer._operation_matches_bound_tenant(entry.operation):
            raise RedoIntegrityError("durable redo entry crosses its user or tenant boundary")
        if canonical_json(entry.operation.to_dict()) != canonical_json(operation.to_dict()):
            raise RedoIntegrityError("redo recovery operation does not match its durable entry")
        if phase != entry.phase:
            raise RedoIntegrityError("redo recovery phase does not match its durable entry")
        if source_effect is not None and canonical_json(source_effect) != canonical_json(entry.source_effect):
            raise RedoIntegrityError("redo Source effect does not match its durable entry")
        if relation_manifest is not None and canonical_json(relation_manifest) != canonical_json(
            entry.relation_manifest
        ):
            raise RedoIntegrityError("redo Relation manifest does not match its durable entry")
        return entry

    @staticmethod
    def _validate_canonical_artifact_keys(committer, operation: ContextOperation) -> tuple[str, str]:
        transaction_id = require_safe_path_segment(
            operation.payload.get("transaction_id"),
            "canonical transaction_id",
        )
        idempotency_key = require_safe_path_segment(
            operation.payload.get("idempotency_key"),
            "canonical idempotency_key",
        )
        return transaction_id, idempotency_key

    @staticmethod
    def _reject_cross_boundary_redo_collisions(
        committer,
        user_id: str,
        operations: list[ContextOperation],
    ) -> None:
        requested_ids = {operation.operation_id for operation in operations}
        if not requested_ids:
            return
        for entry in committer.redo.pending_entries():
            if entry.operation_id not in requested_ids:
                continue
            if entry.operation.user_id != user_id or not committer._operation_matches_bound_tenant(entry.operation):
                raise RedoIntegrityError("redo operation id is already bound to another user or tenant")
            requested = next(item for item in operations if item.operation_id == entry.operation_id)
            if bool(entry.operation.payload.get("canonical_memory")) != bool(
                requested.payload.get("canonical_memory")
            ) or not committer._redo_request_matches_durable_effect(entry.operation, requested):
                raise RedoIntegrityError("redo operation id is bound to a different durable effect")

    @staticmethod
    def _redo_request_matches_durable_effect(
        committer,
        durable: ContextOperation,
        requested: ContextOperation,
    ) -> bool:
        """Compare caller intent with the resolver-bound durable operation.

        A regular UPDATE/MERGE/DELETE may legitimately omit ``target_uri``;
        TargetResolver then binds it from a declared payload URI or a scoped
        candidate.  A retry recreates the original pre-resolution request,
        while the redo entry necessarily stores the resolved target.  Treat
        that difference as equivalent only after rebinding a copy of the
        exact caller payload to the durable target.  An explicitly supplied
        different target remains a hard cross-boundary collision.
        """

        if bool(durable.payload.get("canonical_memory")):
            return committer._operation_effect_fingerprint(durable) == committer._operation_effect_fingerprint(requested)
        if requested.target_uri is not None and requested.target_uri != durable.target_uri:
            return False
        rebound = ContextOperation.from_dict(requested.to_dict())
        rebound.target_uri = durable.target_uri
        return committer._operation_effect_fingerprint(durable) == committer._operation_effect_fingerprint(rebound)
