"""Validation for regular operation effects before durable mutation."""

from __future__ import annotations

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.core.integrity import canonical_json
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


class RegularOperationValidator:
    """Validate generic operation payloads and durable in-flight effects."""

    @staticmethod
    def _validate_regular_operation_effect(
        committer,
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
        if operation.action in policy_actions:
            committer._validate_action_policy_operation(operation)

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
                current_target = committer.source_store.read_object(operation.target_uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                current_target = None
        committer._validate_regular_canonical_boundary(
            operation,
            current_target,
            desired_obj,
            allow_existing_add=allow_existing_add,
        )
        if validate_target_state and operation.action in target_actions:
            if not operation.target_uri:
                raise ValueError(f"{operation.action.value} operation requires a target URI")
            target = committer.source_store.read_object(operation.target_uri)
            if target.context_type != operation.context_type:
                raise ValueError("regular operation target context type mismatch")
            if operation.action in policy_actions and operation.context_type == ContextType.ACTION_POLICY:
                committer._read_action_policy(operation.target_uri)

    @staticmethod
    def _trusted_inflight_regular_object_effect(committer, operation: ContextOperation) -> ContextOperation | None:
        if operation.action not in {OperationAction.ADD, OperationAction.UPDATE, OperationAction.MERGE}:
            return None
        matches = [
            entry
            for entry in committer.redo.pending_entries()
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
        if committer._operation_effect_fingerprint(persisted) != committer._operation_effect_fingerprint(requested):
            raise ValueError("regular redo operation conflicts with the requested effect")
        if not requested.target_uri:
            raise ValueError("regular redo operation is missing its persisted target")
        current = committer.source_store.read_object(requested.target_uri)
        expected_payload = committer._normalized_regular_object_effect(requested)
        if not isinstance(expected_payload, dict) or canonical_json(current.to_dict()) != canonical_json(
            expected_payload
        ):
            raise ValueError("regular redo SourceStore effect does not match its operation")
        expected_content = str(requested.payload.get("content", ""))
        if expected_content or requested.action == OperationAction.ADD:
            try:
                actual_content = committer.source_store.read_content(current.layers.l2_uri or current.uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                actual_content = ""
            if actual_content != expected_content:
                raise ValueError("regular redo SourceStore content does not match its operation")
        return requested
