"""事务目标的显式解析与安全边界校验。

这里只处理调用方已经声明的目标，以及新增对象自身携带的 URI。依赖索引的
模糊检索属于 Context 基础设施能力，由运行时注入具体解析器。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from infrastructure.store.contracts.index import IndexHit
from infrastructure.store.contracts.source import SourceStore
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_uri import ContextURI
from infrastructure.store.model.context.lifecycle import LifecycleState
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_action import OperationAction
from transaction.model.operation_status import OperationStatus


@dataclass(frozen=True)
class ResolveResult:
    operation: ContextOperation
    resolved: bool
    reason: str = ""
    candidates: list[IndexHit] = field(default_factory=list)


class TargetResolver:
    """校验普通 SourceStore 对象的显式事务目标。

    即使自定义 SourceStore 能够接受 Markdown 文档 URI，它也不能成为这里的
    合法目标；记忆文档必须走自己的提交事务。缺少目标时默认转入人工复核，
    不在事务内核中发起检索。
    """

    def __init__(
        self,
        source_store: SourceStore | None = None,
    ) -> None:
        self.source_store = source_store

    def resolve(
        self,
        operation: ContextOperation,
        user_id: str | None = None,
        limit: int = 5,
    ) -> ResolveResult:
        commit_user = str(user_id or operation.user_id)
        if operation.user_id != commit_user:
            return self._reject(operation, "operation_user_mismatch")
        if not isinstance(operation.payload, dict):
            return self._reject(operation, "invalid_operation_payload")
        tenant_id = self._tenant(operation)
        if not tenant_id:
            return self._reject(operation, "invalid_payload_tenant")
        payload_error = self._validate_object_payload(operation, commit_user, tenant_id)
        if payload_error:
            return self._reject(operation, payload_error)

        if not operation.target_uri:
            declared = self._declared_target(operation)
            if declared:
                operation.target_uri = declared
        if operation.target_uri:
            return self._resolve_explicit(
                operation,
                commit_user=commit_user,
                tenant_id=tenant_id,
                creating=operation.action == OperationAction.ADD,
            )

        if operation.action == OperationAction.ADD:
            return self._reject(operation, "payload_target_uri_missing")
        return self._resolve_missing(
            operation,
            commit_user=commit_user,
            tenant_id=tenant_id,
            limit=limit,
        )

    def _resolve_missing(
        self,
        operation: ContextOperation,
        *,
        commit_user: str,
        tenant_id: str,
        limit: int,
    ) -> ResolveResult:
        """缺少目标时保持事务关闭；Context 基础设施可覆盖此扩展点。"""

        del commit_user, tenant_id, limit
        return self._pending(operation, "target_review_required")

    def _resolve_explicit(
        self,
        operation: ContextOperation,
        *,
        commit_user: str,
        tenant_id: str,
        creating: bool,
    ) -> ResolveResult:
        uri = str(operation.target_uri or "")
        error = self._validate_uri(uri, commit_user)
        if error:
            return self._reject(operation, error)
        declared = self._declared_target(operation)
        if operation.action != OperationAction.SUPERSEDE and declared and declared != uri:
            return self._reject(operation, "payload_target_uri_mismatch")
        if operation.action == OperationAction.SUPERSEDE:
            replacement = self._object_uri(operation)
            if not replacement:
                return self._reject(operation, "payload_target_uri_missing")
            if replacement == uri:
                return self._reject(operation, "supersede_replacement_matches_target")
            replacement_error = self._validate_uri(replacement, commit_user)
            if replacement_error:
                return self._reject(operation, f"payload_{replacement_error}")
        if not creating:
            if self.source_store is None:
                return self._pending(operation, "target_validation_unavailable")
            try:
                target = self.source_store.read_object(uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                return self._reject(operation, "target_not_found")
            except (TypeError, ValueError):
                return self._reject(operation, "target_object_invalid")
            object_error = self._validate_target(target, operation, commit_user, tenant_id)
            if object_error:
                return self._reject(operation, object_error)
        operation.status = OperationStatus.RESOLVED
        return ResolveResult(operation, True, "explicit target validated")

    @staticmethod
    def _object_uri(operation: ContextOperation) -> str:
        payload = operation.payload.get("context_object")
        return str(payload.get("uri") or "") if isinstance(payload, dict) else ""

    def _declared_target(self, operation: ContextOperation) -> str:
        if operation.action == OperationAction.ADD:
            return self._object_uri(operation)
        for value in (
            operation.payload.get("target_uri"),
            operation.payload.get("policy_uri"),
            self._object_uri(operation) if operation.action != OperationAction.SUPERSEDE else "",
        ):
            if isinstance(value, str) and value:
                return value
        return ""

    def _validate_object_payload(
        self,
        operation: ContextOperation,
        commit_user: str,
        tenant_id: str,
    ) -> str:
        raw = operation.payload.get("context_object")
        if raw is None:
            return ""
        if not isinstance(raw, dict):
            return "invalid_payload_context_object"
        if raw.get("owner_user_id") not in {None, "", commit_user}:
            return "payload_owner_mismatch"
        if raw.get("tenant_id") not in {None, "", tenant_id}:
            return "payload_tenant_mismatch"
        if raw.get("context_type") not in {None, "", operation.context_type.value}:
            return "payload_context_type_mismatch"
        uri = raw.get("uri")
        if uri is not None and not isinstance(uri, str):
            return "invalid_payload_target_uri"
        if isinstance(uri, str) and uri:
            error = self._validate_uri(uri, commit_user)
            if error:
                return f"payload_{error}"
        return ""

    @staticmethod
    def _validate_uri(uri: str, commit_user: str) -> str:
        try:
            parsed = ContextURI.parse(uri)
        except (TypeError, ValueError):
            return "invalid_target_uri"
        if parsed.authority == "user" and parsed.user_id != commit_user:
            return "target_owner_mismatch"
        if parsed.authority == "user" and parsed.segments[1:3] == ("memory", "documents"):
            return "document_target_forbidden"
        return ""

    @staticmethod
    def _validate_target(
        target: ContextObject,
        operation: ContextOperation,
        commit_user: str,
        tenant_id: str,
        *,
        expected_uri: str | None = None,
    ) -> str:
        if target.uri != (expected_uri or operation.target_uri):
            return "target_uri_mismatch"
        if target.owner_user_id not in {None, "", commit_user}:
            return "target_owner_mismatch"
        if str(target.tenant_id or "default") != tenant_id:
            return "target_tenant_mismatch"
        if target.context_type != operation.context_type:
            return "target_context_type_mismatch"
        if target.lifecycle_state in {
            LifecycleState.DELETED,
            LifecycleState.ARCHIVED,
            LifecycleState.OBSOLETE,
        }:
            return "target_not_active"
        return ""

    def _tenant(self, operation: ContextOperation) -> str:
        values = []
        raw = operation.payload.get("tenant_id")
        if raw not in {None, ""}:
            values.append(raw)
        obj = operation.payload.get("context_object")
        if isinstance(obj, dict) and obj.get("tenant_id") not in {None, ""}:
            values.append(obj["tenant_id"])
        if any(not isinstance(value, str) or not value.strip() for value in values):
            return ""
        if len(set(values)) > 1:
            return ""
        return str(values[0]) if values else str(getattr(self.source_store, "tenant_id", "default") or "default")

    @staticmethod
    def _pending(
        operation: ContextOperation,
        reason: str,
        candidates: list[IndexHit] | None = None,
    ) -> ResolveResult:
        operation.status = OperationStatus.PENDING
        operation.payload["target_resolution_reason"] = reason
        selected = list(candidates or [])
        if selected:
            operation.payload["target_candidates"] = [hit.__dict__ for hit in selected]
        return ResolveResult(operation, False, reason, selected)

    @staticmethod
    def _reject(operation: ContextOperation, reason: str) -> ResolveResult:
        operation.status = OperationStatus.REJECTED
        operation.payload["target_resolution_reason"] = reason
        return ResolveResult(operation, False, reason)


__all__ = ["ResolveResult", "TargetResolver"]
