from __future__ import annotations

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction
from memoryos.operations.model.operation_status import OperationStatus
from memoryos.operations.resolver.target_resolver import TargetResolver


def _operation(*, target_uri: str | None = None, payload: dict | None = None) -> ContextOperation:
    return ContextOperation(
        user_id="u1",
        context_type=ContextType.BEHAVIOR_CASE,
        action=OperationAction.UPDATE,
        target_uri=target_uri,
        payload=payload or {},
    )


def test_explicit_target_requires_matching_owner_tenant_and_type(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path, tenant_id="tenant-a")
    index = InMemoryIndexStore()
    target = ContextObject(
        uri="memoryos://user/u1/behavior_cases/a",
        context_type=ContextType.BEHAVIOR_CASE,
        title="a",
        owner_user_id="u1",
        tenant_id="tenant-a",
    )
    source.write_object(target, content="alpha")
    resolver = TargetResolver(index, source_store=source)

    accepted = resolver.resolve(
        _operation(target_uri=target.uri, payload={"tenant_id": "tenant-a"}),
        user_id="u1",
    )
    wrong_tenant = resolver.resolve(
        _operation(target_uri=target.uri, payload={"tenant_id": "tenant-b"}),
        user_id="u1",
    )
    wrong_owner = resolver.resolve(
        _operation(target_uri="memoryos://user/u2/behavior_cases/a", payload={"tenant_id": "tenant-a"}),
        user_id="u1",
    )

    assert accepted.resolved
    assert wrong_tenant.operation.status == OperationStatus.REJECTED
    assert wrong_owner.reason == "target_owner_mismatch"


def test_add_binds_context_object_uri_without_search(tmp_path) -> None:  # noqa: ANN001
    obj = ContextObject(
        uri="memoryos://user/u1/behavior_cases/new",
        context_type=ContextType.BEHAVIOR_CASE,
        title="new",
        owner_user_id="u1",
    )
    operation = ContextOperation(
        user_id="u1",
        context_type=ContextType.BEHAVIOR_CASE,
        action=OperationAction.ADD,
        payload={"context_object": obj.to_dict()},
    )

    result = TargetResolver(InMemoryIndexStore(), source_store=FileSystemSourceStore(tmp_path)).resolve(
        operation,
        user_id="u1",
    )

    assert result.resolved
    assert result.operation.target_uri == obj.uri


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
    result = TargetResolver(index, source_store=source).resolve(
        _operation(payload={"query": "alpha behavior", "context_object": desired.to_dict()}),
        user_id="u1",
    )

    assert result.resolved
    assert result.operation.target_uri == wanted.uri


def test_document_uri_is_never_a_target(tmp_path) -> None:  # noqa: ANN001
    uri = "memoryos://user/u1/memory/documents/018f47c0-7c55-7b09-8f6e-123456789abc"
    result = TargetResolver(InMemoryIndexStore(), source_store=FileSystemSourceStore(tmp_path)).resolve(
        ContextOperation(
            user_id="u1",
            context_type=ContextType.MEMORY,
            action=OperationAction.DELETE,
            target_uri=uri,
            payload={},
        ),
        user_id="u1",
    )

    assert not result.resolved
    assert result.reason == "document_target_forbidden"
    assert result.operation.status == OperationStatus.REJECTED
