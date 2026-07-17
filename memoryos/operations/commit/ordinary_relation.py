"""Operations-owned command construction for an ordinary relation update."""

from __future__ import annotations

from typing import TYPE_CHECKING

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction

if TYPE_CHECKING:
    from memoryos.operations.commit.operation_committer import OperationCommitter
    from memoryos.operations.model.context_diff import ContextDiff


def commit_ordinary_relation_update(
    committer: OperationCommitter,
    *,
    owner_user_id: str,
    desired_authority: ContextObject,
    content: str,
    tenant_id: str,
) -> ContextDiff:
    """Build and commit the operation backing a ContextDB relation mutation."""

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
