from __future__ import annotations

from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore, InMemoryRelationStore
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


def _operation() -> ContextOperation:
    obj = ContextObject(uri="memoryos://user/u1/memories/profile/a", context_type=ContextType.MEMORY, title="a", owner_user_id="u1")
    return ContextOperation(
        user_id="u1",
        context_type=ContextType.MEMORY,
        action=OperationAction.ADD,
        target_uri=obj.uri,
        payload={"context_object": obj.to_dict(), "content": "alpha"},
    )


def test_commit_operation_calls_operation_committer(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    db = ContextDB(source, index, InMemoryRelationStore(), committer=OperationCommitter(source, index, str(tmp_path)))

    result = db.commit_operation(_operation())

    assert len(result.operations) == 1
    assert source.read_object("memoryos://user/u1/memories/profile/a").title == "a"
    assert index.search("alpha", filters={"owner_user_id": "u1"})


def test_commit_operation_requires_committer(tmp_path) -> None:
    db = ContextDB(FileSystemSourceStore(tmp_path), InMemoryIndexStore(), InMemoryRelationStore())

    try:
        db.commit_operation(_operation())
    except RuntimeError as exc:
        assert "OperationCommitter" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("commit_operation should require OperationCommitter")


def test_seed_object_directly_writes_source_and_index_for_seed_data(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    db = ContextDB(source, index, InMemoryRelationStore())
    obj = ContextObject(uri="memoryos://user/u1/memories/profile/seed", context_type=ContextType.MEMORY, title="seed", owner_user_id="u1")

    db.seed_object(obj, content="seed content")

    assert source.read_object(obj.uri).title == "seed"
    assert index.search("seed", filters={"owner_user_id": "u1"})
