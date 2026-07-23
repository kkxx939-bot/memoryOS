"""在耐久写入前校验普通操作及其预期副作用。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from foundation.integrity import canonical_json
from infrastructure.store.model.context.context_object import ContextObject
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_action import OperationAction

if TYPE_CHECKING:
    from transaction.commit.host import OperationTransactionHost


class RegularOperationValidator:
    """校验通用操作载荷和处于事务中的耐久副作用。"""

    def _validate_regular_operation_effect(
        self: OperationTransactionHost,
        operation: ContextOperation,
        *,
        validate_target_state: bool,
        allow_existing_add: bool = False,
    ) -> None:
        if not isinstance(operation.payload, dict):
            raise ValueError("regular operation payload must be an object")
        # 第一次修改 SourceStore 前，必须确认 Redo、审计、Diff 和幂等标记
        # 需要写入的操作结构都能够稳定序列化。
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

        domain_handled = self._validate_domain_operation(operation)
        generic_actions = {
            OperationAction.ADD,
            OperationAction.UPDATE,
            OperationAction.MERGE,
            OperationAction.DELETE,
            OperationAction.ARCHIVE,
            OperationAction.COMPRESS,
            OperationAction.REFRESH_LAYERS,
            OperationAction.REINDEX,
            OperationAction.SUPERSEDE,
        }
        if operation.action not in generic_actions and not domain_handled:
            raise ValueError(f"{operation.action.value} operation requires an injected domain handler")

        target_actions = {
            OperationAction.UPDATE,
            OperationAction.MERGE,
            OperationAction.DELETE,
            OperationAction.ARCHIVE,
            OperationAction.COMPRESS,
            OperationAction.REFRESH_LAYERS,
            OperationAction.REINDEX,
            OperationAction.SUPERSEDE,
        }
        if domain_handled:
            target_actions.add(operation.action)
        current_target: ContextObject | None = None
        if operation.target_uri:
            try:
                current_target = self.source_store.read_object(operation.target_uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                current_target = None
        if desired_obj is not None:
            if desired_obj.owner_user_id not in {None, "", operation.user_id}:
                raise ValueError("regular operation context_object owner mismatch")
            if str(desired_obj.tenant_id or "default") != self.tenant_id:
                raise ValueError("regular operation context_object tenant mismatch")
        if operation.action == OperationAction.ADD and current_target is not None and not allow_existing_add:
            raise ValueError("add operation target already exists")
        if validate_target_state and operation.action in target_actions:
            if not operation.target_uri:
                raise ValueError(f"{operation.action.value} operation requires a target URI")
            target = self.source_store.read_object(operation.target_uri)
            if target.context_type != operation.context_type:
                raise ValueError("regular operation target context type mismatch")

    def _trusted_inflight_regular_object_effect(
        self: OperationTransactionHost, operation: ContextOperation
    ) -> ContextOperation | None:
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
