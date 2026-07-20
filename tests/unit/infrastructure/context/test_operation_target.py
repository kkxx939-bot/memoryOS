"""Context 索引目标搜索的范围与置信度测试。"""

from __future__ import annotations

from infrastructure.context.operation_target import ContextOperationTargetResolver
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_type import ContextType
from tests.support.persistence import FileSystemSourceStore, InMemoryIndexStore
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_action import OperationAction


def test_automatic_resolution_is_context_type_scoped(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    wanted = ContextObject(
        uri="memoryos://user/u1/behavior_cases/alpha",
        context_type=ContextType.BEHAVIOR_CASE,
        title="alpha behavior",
        owner_user_id="u1",
    )
    other = ContextObject(
        uri="memoryos://user/u1/action_policies/alpha",
        context_type=ContextType.ACTION_POLICY,
        title="alpha policy",
        owner_user_id="u1",
    )
    for obj in (wanted, other):
        source.write_object(obj, content=obj.title)
        index.upsert_index(obj, content=obj.title, tenant_id="default")
    desired = ContextObject.from_dict(wanted.to_dict())
    desired.title = "updated"
    operation = ContextOperation(
        user_id="u1",
        context_type=ContextType.BEHAVIOR_CASE,
        action=OperationAction.UPDATE,
        payload={"query": "alpha behavior", "context_object": desired.to_dict()},
    )

    result = ContextOperationTargetResolver(index, source).resolve(operation, user_id="u1")

    assert result.resolved
    assert result.operation.target_uri == wanted.uri
