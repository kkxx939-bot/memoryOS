"""把普通关系语义转换为通用操作事务命令。"""

from __future__ import annotations

from typing import Any

from infrastructure.context.commit_protocol import OrdinaryRelationCommitter
from infrastructure.store.model.context.context_object import ContextObject
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_action import OperationAction


def commit_ordinary_relation_update(
    committer: OrdinaryRelationCommitter,
    *,
    owner_user_id: str,
    desired_authority: ContextObject,
    content: str,
    tenant_id: str,
) -> Any:
    """由 Context 构造关系更新命令，再交给通用事务层耐久执行。"""

    if committer.tenant_id != tenant_id:
        raise RuntimeError("ordinary relation committer differs from its Source tenant")
    operation = ContextOperation(
        user_id=owner_user_id,
        context_type=desired_authority.context_type,
        action=OperationAction.UPDATE,
        target_uri=desired_authority.uri,
        payload={
            "tenant_id": tenant_id,
            "context_object": desired_authority.to_dict(),
            "content": content,
            "reason": "context_relation_upsert",
        },
    )
    return committer.commit(owner_user_id, [operation])


__all__ = ["commit_ordinary_relation_update"]
