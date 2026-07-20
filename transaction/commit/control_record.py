"""把操作结果投影成不含语义正文的控制记录。"""

from __future__ import annotations

from collections.abc import Callable

from transaction.model.context_diff import ContextDiff
from transaction.model.context_operation import ContextOperation

OperationFingerprint = Callable[[ContextOperation], str]


def operation_control_record(
    operation: ContextOperation,
    *,
    tenant_id: str,
    fingerprint: OperationFingerprint,
) -> dict:
    """只保存恢复校验和幂等判断需要的稳定字段。"""

    return {
        "operation_id": operation.operation_id,
        "user_id": operation.user_id,
        "tenant_id": tenant_id,
        "context_type": operation.context_type.value,
        "action": operation.action.value,
        "target_uri": operation.target_uri,
        "status": operation.status.value,
        "created_at": operation.created_at,
        "effect_fingerprint": fingerprint(operation),
    }


def diff_control_record(
    diff: ContextDiff,
    *,
    tenant_id: str,
    fingerprint: OperationFingerprint,
) -> dict:
    """把公开 ContextDiff 投影为不含 payload 和 evidence 的耐久索引。"""

    return {
        "diff_id": diff.diff_id,
        "user_id": diff.user_id,
        "tenant_id": tenant_id,
        "created_at": diff.created_at,
        "schema_version": "context_diff_control_v1",
        "operations": [
            operation_control_record(item, tenant_id=tenant_id, fingerprint=fingerprint) for item in diff.operations
        ],
        "pending_operations": [
            operation_control_record(item, tenant_id=tenant_id, fingerprint=fingerprint)
            for item in diff.pending_operations
        ],
        "rejected_operations": [
            operation_control_record(item, tenant_id=tenant_id, fingerprint=fingerprint)
            for item in diff.rejected_operations
        ],
    }


def diff_control_members(payload: dict) -> tuple[tuple[str, str, str, str], ...]:
    """读取控制记录中的成员身份，用于幂等冲突校验。"""

    members: list[tuple[str, str, str, str]] = []
    for kind in ("operations", "pending_operations", "rejected_operations"):
        raw_items = payload.get(kind, [])
        if not isinstance(raw_items, list):
            raise ValueError("diff control members must be a list")
        for item in raw_items:
            if not isinstance(item, dict):
                raise ValueError("diff control member must be an object")
            operation_id = str(item.get("operation_id") or "")
            status = str(item.get("status") or "")
            effect_fingerprint = str(item.get("effect_fingerprint") or "")
            if not operation_id or not status or not effect_fingerprint:
                raise ValueError("diff control member is incomplete")
            members.append((kind, operation_id, status, effect_fingerprint))
    return tuple(sorted(members))


__all__ = [
    "diff_control_members",
    "diff_control_record",
    "operation_control_record",
]
