"""Shared safety rules for ordinary operation commits and recovery."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Protocol, cast

from memoryos.contextdb.model.context_uri import ContextURI
from memoryos.core.ids import require_safe_path_segment
from memoryos.core.integrity import canonical_json
from memoryos.operations.commit.redo_log import RedoEntry, RedoIntegrityError
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class _TombstoneRunResult(Protocol):
    processed: tuple[str, ...]
    failed: tuple[str, ...]
    stale: tuple[str, ...]


class CommitStateMachine:
    @staticmethod
    def _delete_tombstone_ids(operation: ContextOperation) -> tuple[str, ...]:
        raw = operation.payload.get("projection_tombstone_ids", ())
        if not isinstance(raw, list | tuple) or any(not isinstance(item, str) or not item for item in raw):
            return ()
        return tuple(dict.fromkeys(raw))

    @staticmethod
    def _require_delete_tombstone_capability(committer, operations: list[ContextOperation]) -> None:
        if not any(operation.action == OperationAction.DELETE for operation in operations):
            return
        if callable(getattr(committer.index_store, "enqueue_tombstone", None)) and committer.tombstone_service is None:
            raise RuntimeError("production DELETE requires ProjectionTombstoneService")

    @staticmethod
    def _prepare_delete_tombstones(
        committer,
        operation: ContextOperation,
        *,
        trust_durable_binding: bool = False,
    ) -> tuple[str, ...]:
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
            reason=str(operation.payload.get("reason") or operation.action.value),
            require_source_retired=True,
        )
        if not isinstance(raw_ids, list | tuple):
            raise RuntimeError("durable projection tombstone journal returned an invalid result")
        ids = tuple(dict.fromkeys(str(item) for item in raw_ids if str(item)))
        if not ids:
            raise RuntimeError("production DELETE did not journal a projection tombstone")
        operation.payload["projection_tombstone_ids"] = list(ids)
        return ids

    @staticmethod
    def _settle_delete_tombstones(committer, operations: list[ContextOperation]) -> None:
        if committer.tombstone_service is None:
            return
        process = getattr(committer.tombstone_service, "process_tombstones", None)
        if not callable(process):
            raise TypeError("ProjectionTombstoneService has no exact replay operation")
        for operation in operations:
            if operation.action != OperationAction.DELETE:
                continue
            ids = committer._delete_tombstone_ids(operation)
            if not ids:
                raise RuntimeError("committed production DELETE has no durable tombstone binding")
            result = cast(
                _TombstoneRunResult,
                process(ids, tenant_id=committer.tenant_id),
            )
            if any(getattr(result, name, None) is None for name in ("failed", "processed", "stale")):
                raise RuntimeError("durable projection tombstone cleanup returned an invalid result")
            if result.failed:
                raise RuntimeError("derived projection tombstone cleanup is retryable but incomplete")

    @staticmethod
    @contextmanager
    def _durable_startup_recovery_scope(committer, group_id: str) -> Iterator[None]:
        require_safe_path_segment(group_id, "startup recovery commit_group_id")
        readiness = getattr(committer.source_store, "readiness", None)
        state = str(getattr(getattr(readiness, "state", None), "value", ""))
        if state and state != "RECOVERING":
            raise RuntimeError("durable startup commit scope requires a RECOVERING runtime")
        token = committer._startup_recovery_group.set(group_id)
        try:
            yield
        finally:
            committer._startup_recovery_group.reset(token)

    @staticmethod
    def _require_commit_ready(committer, user_id: str, operations: list[ContextOperation]) -> None:
        del user_id, operations
        readiness = getattr(committer.source_store, "readiness", None)
        require_ready = getattr(readiness, "require_ready", None)
        if not callable(require_ready) or committer._startup_recovery_group.get():
            return
        require_ready()

    @staticmethod
    def _notify(committer, stage: str, operation_id: str) -> None:
        if callable(committer.test_hook):
            committer.test_hook(stage, operation_id)

    @staticmethod
    def _lock_key(committer, raw_key: str) -> str:
        key = str(ContextURI.parse(raw_key)) if raw_key.startswith("memoryos://") else raw_key
        return key if committer.tenant_id == "default" else f"tenant:{committer.tenant_id}:{key}"

    @staticmethod
    def _validate_tenant_id(value: object, label: str) -> str:
        if not isinstance(value, str) or not value.strip() or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError(f"{label} must be one safe non-empty path segment")
        return value

    @staticmethod
    def _explicit_tenant_declarations(committer, operation: ContextOperation) -> list[tuple[str, str]]:
        if not isinstance(operation.payload, dict):
            raise ValueError("operation payload must be an object")
        declarations: list[tuple[str, str]] = []

        def inspect(value: object, path: str) -> None:
            if isinstance(value, dict) and "tenant_id" in value:
                declarations.append(
                    (path, committer._validate_tenant_id(value["tenant_id"], f"{path}.tenant_id"))
                )

        def walk(value: object, path: str) -> None:
            if not isinstance(value, dict):
                return
            inspect(value, path)
            for name in ("scope", "visibility", "authority"):
                nested = value.get(name)
                inspect(nested, f"{path}.{name}")
            metadata = value.get("metadata")
            if isinstance(metadata, dict):
                walk(metadata, f"{path}.metadata")

        walk(operation.payload, "payload")
        walk(operation.payload.get("context_object"), "payload.context_object")
        return declarations

    @staticmethod
    def _operation_matches_bound_tenant(committer, operation: ContextOperation) -> bool:
        try:
            return all(value == committer.tenant_id for _, value in committer._explicit_tenant_declarations(operation))
        except ValueError:
            return False

    @staticmethod
    def _validate_and_bind_operations(
        committer,
        user_id: str,
        operations: list[ContextOperation],
    ) -> None:
        require_safe_path_segment(user_id, "commit user_id")
        for operation in operations:
            require_safe_path_segment(operation.operation_id, "operation_id")
            if operation.user_id != user_id:
                raise ValueError("operation user does not match commit user")
            declarations = committer._explicit_tenant_declarations(operation)
            if any(value != committer.tenant_id for _, value in declarations):
                paths = ", ".join(path for path, value in declarations if value != committer.tenant_id)
                raise ValueError(f"operation tenant does not match bound tenant: {paths}")
            committer._reject_document_owned_operation(operation)
        for operation in operations:
            operation.payload.setdefault("tenant_id", committer.tenant_id)
            obj = operation.payload.get("context_object")
            if isinstance(obj, dict):
                obj.setdefault("tenant_id", committer.tenant_id)

    @staticmethod
    def _validate_recovery_artifact_tenant(committer, payload: object, label: str) -> None:
        if not isinstance(payload, dict) or "tenant_id" not in payload:
            return
        if committer._validate_tenant_id(payload["tenant_id"], f"{label} tenant_id") != committer.tenant_id:
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
        except (PermissionError, ValueError) as exc:
            raise RedoIntegrityError("redo operation crosses its ordinary user, tenant, or document boundary") from exc
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
        matches = [entry for entry in committer.redo.pending_entries() if entry.operation_id == operation.operation_id]
        if len(matches) != 1:
            raise RedoIntegrityError("redo recovery requires exactly one durable operation entry")
        entry = matches[0]
        if entry.user_id != user_id or not committer._operation_matches_bound_tenant(entry.operation):
            raise RedoIntegrityError("durable redo entry crosses its user or tenant boundary")
        if canonical_json(entry.operation.to_dict()) != canonical_json(operation.to_dict()):
            raise RedoIntegrityError("redo recovery operation does not match its durable entry")
        if entry.phase != phase:
            raise RedoIntegrityError("redo recovery phase does not match its durable entry")
        if source_effect is not None and canonical_json(source_effect) != canonical_json(entry.source_effect):
            raise RedoIntegrityError("redo Source effect does not match its durable entry")
        if relation_manifest is not None and canonical_json(relation_manifest) != canonical_json(entry.relation_manifest):
            raise RedoIntegrityError("redo Relation manifest does not match its durable entry")
        return entry

    @staticmethod
    def _reject_cross_boundary_redo_collisions(
        committer,
        user_id: str,
        operations: list[ContextOperation],
    ) -> None:
        requested = {operation.operation_id: operation for operation in operations}
        for entry in committer.redo.pending_entries():
            candidate = requested.get(entry.operation_id)
            if candidate is None:
                continue
            if entry.user_id != user_id or not committer._operation_matches_bound_tenant(entry.operation):
                raise RedoIntegrityError("redo operation id is already bound to another user or tenant")
            if not committer._redo_request_matches_durable_effect(entry.operation, candidate):
                raise RedoIntegrityError("redo operation id is bound to a different durable effect")

    @staticmethod
    def _redo_request_matches_durable_effect(
        committer,
        durable: ContextOperation,
        requested: ContextOperation,
    ) -> bool:
        if requested.target_uri is not None and requested.target_uri != durable.target_uri:
            return False
        rebound = ContextOperation.from_dict(requested.to_dict())
        rebound.target_uri = durable.target_uri
        return committer._operation_effect_fingerprint(durable) == committer._operation_effect_fingerprint(rebound)


__all__ = ["CommitStateMachine"]
