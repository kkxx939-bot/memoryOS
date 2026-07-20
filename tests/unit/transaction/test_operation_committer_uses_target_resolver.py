"""事务提交器与可注入目标解析器的协作测试。"""

from __future__ import annotations

from infrastructure.context.operation_effects import InfrastructureContextOperationEffects
from infrastructure.context.operation_target import ContextOperationTargetResolver
from infrastructure.store.contracts.index import IndexHit
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_type import ContextType
from infrastructure.store.model.context.lifecycle import LifecycleState
from tests.support.persistence import FileSystemSourceStore, InMemoryIndexStore
from tests.support.transaction import build_test_operation_committer as OperationCommitter
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_action import OperationAction
from transaction.model.operation_status import OperationStatus


def test_update_without_target_resolves_before_apply(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    old = ContextObject(
        uri="memoryos://user/u1/behavior_cases/temperature",
        context_type=ContextType.BEHAVIOR_CASE,
        title="temperature",
        owner_user_id="u1",
    )
    source.write_object(old, content="old")
    index.upsert_index(old, content="prefers 26 degree", tenant_id="default")
    updated = ContextObject.from_dict(old.to_dict())
    updated.title = "temperature updated"
    operation = ContextOperation(
        user_id="u1",
        context_type=ContextType.BEHAVIOR_CASE,
        action=OperationAction.UPDATE,
        payload={
            "query": "prefers 26 degree",
            "context_object": updated.to_dict(),
            "content": "new",
        },
    )

    diff = OperationCommitter(
        source,
        index,
        str(tmp_path),
        target_resolver=ContextOperationTargetResolver(index, source),
    ).commit("u1", [operation])

    assert diff.operations[0].target_uri == old.uri
    assert source.read_content(old.uri) == "new"


def test_delete_action_policy_without_target_resolves_by_scene_and_action(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    uri = "memoryos://user/u1/action_policies/hot_room/turn_on_ac"
    obj = ContextObject(
        uri=uri,
        context_type=ContextType.ACTION_POLICY,
        title="hot_room turn_on_ac",
        owner_user_id="u1",
    )
    source.write_object(obj, content="policy")
    index.upsert_index(obj, content="hot_room turn_on_ac", tenant_id="default")
    operation = ContextOperation(
        user_id="u1",
        context_type=ContextType.ACTION_POLICY,
        action=OperationAction.DELETE,
        payload={"scene_key": "hot_room", "action": "turn_on_ac"},
    )

    diff = OperationCommitter(
        source,
        index,
        str(tmp_path),
        target_resolver=ContextOperationTargetResolver(index, source),
    ).commit("u1", [operation])

    assert diff.operations[0].target_uri == uri
    assert source.read_object(uri).lifecycle_state == LifecycleState.DELETED
    assert not index.search(
        "turn_on_ac",
        tenant_id="default",
        filters={"owner_user_id": "u1", "context_type": "action_policy"},
    )


def test_low_confidence_automatic_target_stays_pending(tmp_path) -> None:  # noqa: ANN001
    class LowIndex(InMemoryIndexStore):
        def search(self, query: str, *, tenant_id: str, filters=None, limit: int = 10):  # noqa: ANN001, ANN201
            del query, tenant_id, filters, limit
            return [
                IndexHit(
                    uri="memoryos://user/u1/behavior_cases/old",
                    score=0.2,
                    context_type="behavior_case",
                    title="old",
                    metadata={"retrieval_scores": {"lexical": 0.2, "vector": 0.0, "identity": 0.0}},
                )
            ]

    source = FileSystemSourceStore(tmp_path)
    index = LowIndex()
    old = ContextObject(
        uri="memoryos://user/u1/behavior_cases/old",
        context_type=ContextType.BEHAVIOR_CASE,
        title="old",
        owner_user_id="u1",
    )
    source.write_object(old, content="ambiguous")
    replacement = ContextObject(
        uri="memoryos://user/u1/behavior_cases/new",
        context_type=ContextType.BEHAVIOR_CASE,
        title="new",
        owner_user_id="u1",
    )
    operation = ContextOperation(
        user_id="u1",
        context_type=ContextType.BEHAVIOR_CASE,
        action=OperationAction.SUPERSEDE,
        payload={"query": "ambiguous", "context_object": replacement.to_dict(), "content": "new"},
    )

    diff = OperationCommitter(
        source,
        index,
        str(tmp_path),
        target_resolver=ContextOperationTargetResolver(index, source),
    ).commit("u1", [operation])

    assert not diff.operations
    assert diff.pending_operations[0].status == OperationStatus.PENDING
    assert source.read_object(old.uri).lifecycle_state == LifecycleState.ACTIVE


def test_explicit_cross_user_delete_is_rejected_without_side_effect(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    uri = "memoryos://user/u2/behavior_cases/private"
    target = ContextObject(
        uri=uri,
        context_type=ContextType.BEHAVIOR_CASE,
        title="private",
        owner_user_id="u2",
    )
    source.write_object(target, content="private")
    index.upsert_index(target, content="private", tenant_id="default")
    operation = ContextOperation(
        user_id="u1",
        context_type=ContextType.BEHAVIOR_CASE,
        action=OperationAction.DELETE,
        target_uri=uri,
        payload={"tenant_id": "default"},
    )

    diff = OperationCommitter(source, index, str(tmp_path)).commit("u1", [operation])

    assert not diff.operations
    assert [item.operation_id for item in diff.rejected_operations] == [operation.operation_id]
    assert source.read_object(uri).lifecycle_state == LifecycleState.ACTIVE


def test_document_owned_uri_is_rejected_before_redo_or_source_write(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    uri = "memoryos://user/u1/memory/documents/018f47c0-7c55-7b09-8f6e-123456789abc"
    operation = ContextOperation(
        user_id="u1",
        context_type=ContextType.MEMORY,
        action=OperationAction.DELETE,
        target_uri=uri,
        payload={},
    )

    try:
        OperationCommitter(
            source,
            index,
            str(tmp_path),
            context_effects=InfrastructureContextOperationEffects(),
        ).commit("u1", [operation])
    except PermissionError as exc:
        assert "MemoryDocumentCommitter" in str(exc)
    else:
        raise AssertionError("document-owned URI was accepted")
    assert not (tmp_path / "system" / "redo").exists()


def test_ordinary_typed_relation_cannot_publish_document_endpoint(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    document_uri = "memoryos://user/u1/memory/documents/018f47c0-7c55-7b09-8f6e-123456789abc"
    obj = ContextObject(
        uri="memoryos://user/u1/behavior_patterns/hot-room",
        context_type=ContextType.BEHAVIOR_PATTERN,
        title="hot room",
        owner_user_id="u1",
        metadata={"support_anchor_uri": document_uri},
    )
    operation = ContextOperation(
        user_id="u1",
        context_type=ContextType.BEHAVIOR_PATTERN,
        action=OperationAction.ADD,
        target_uri=obj.uri,
        payload={"context_object": obj.to_dict()},
    )

    try:
        OperationCommitter(
            source,
            index,
            str(tmp_path),
            context_effects=InfrastructureContextOperationEffects(),
        ).commit("u1", [operation])
    except PermissionError as exc:
        assert "MemoryDocumentCommitter" in str(exc)
    else:
        raise AssertionError("ordinary relation targeted a Markdown document")
    assert not (tmp_path / "system").exists()
