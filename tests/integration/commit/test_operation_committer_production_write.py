from __future__ import annotations

from pathlib import Path

from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_type import ContextType
from tests.support.persistence import FileSystemSourceStore, InMemoryIndexStore, seed_context_object
from tests.support.transaction import build_test_operation_committer as OperationCommitter
from transaction.model.context_operation import ContextOperation
from transaction.model.operation_action import OperationAction


def test_production_write_uses_operation_committer_and_test_seed_is_explicit(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    committer = OperationCommitter(source, index, str(tmp_path))
    obj = ContextObject(
        uri="memoryos://user/u1/resources/profile/prod",
        context_type=ContextType.RESOURCE,
        title="prod",
        owner_user_id="u1",
    )
    operation = ContextOperation(
        user_id="u1",
        context_type=ContextType.RESOURCE,
        action=OperationAction.ADD,
        target_uri=obj.uri,
        payload={"context_object": obj.to_dict(), "content": "prod content"},
    )

    committer.commit(operation.user_id, [operation])

    assert (tmp_path / "system" / "audit" / "u1.jsonl").exists()
    assert list((tmp_path / "system" / "diffs").glob("*.json"))

    seed = ContextObject(
        uri="memoryos://user/u1/resources/profile/seed",
        context_type=ContextType.RESOURCE,
        title="seed",
        owner_user_id="u1",
    )
    seed_context_object(source, index, seed, content="seed")
    assert source.read_object(seed.uri).title == "seed"


def test_readme_does_not_show_contextdb_write_object_as_production_update() -> None:
    text = Path("README.md").read_text(encoding="utf-8")
    assert "context_db.write_object" not in text
