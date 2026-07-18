from __future__ import annotations

from pathlib import Path

from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore, InMemoryRelationStore
from memoryos.operations.commit.operation_committer import OperationCommitter
from memoryos.operations.model.context_operation import ContextOperation
from memoryos.operations.model.operation_action import OperationAction


def test_contextdb_production_write_uses_operation_committer_and_seed_is_explicit(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    db = ContextDB(source, index, InMemoryRelationStore(), committer=OperationCommitter(source, index, str(tmp_path)))
    obj = ContextObject(uri="memoryos://user/u1/resources/profile/prod", context_type=ContextType.RESOURCE, title="prod", owner_user_id="u1")
    operation = ContextOperation(user_id="u1", context_type=ContextType.RESOURCE, action=OperationAction.ADD, target_uri=obj.uri, payload={"context_object": obj.to_dict(), "content": "prod content"})

    db.commit_operation(operation)

    assert (tmp_path / "system" / "audit" / "u1.jsonl").exists()
    assert list((tmp_path / "system" / "diffs").glob("*.json"))

    seed = ContextObject(uri="memoryos://user/u1/resources/profile/seed", context_type=ContextType.RESOURCE, title="seed", owner_user_id="u1")
    db.seed_object(seed, content="seed")
    assert source.read_object(seed.uri).title == "seed"


def test_readme_does_not_show_contextdb_write_object_as_production_update() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "context_db.write_object" not in text
