from __future__ import annotations

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore
from memoryos.contextdb.transaction.consistency import ConsistencyVerifier


def test_consistency_reports_missing_orphan_and_deleted_hot_index(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    source_obj = ContextObject(uri="memoryos://user/u1/memories/profile/a", context_type=ContextType.MEMORY, title="a", owner_user_id="u1")
    orphan = ContextObject(uri="memoryos://user/u1/memories/profile/orphan", context_type=ContextType.MEMORY, title="orphan", owner_user_id="u1")
    deleted = ContextObject(uri="memoryos://user/u1/memories/profile/deleted", context_type=ContextType.MEMORY, title="deleted", owner_user_id="u1", lifecycle_state=LifecycleState.DELETED)
    source.write_object(source_obj, content="alpha")
    source.write_object(deleted, content="deleted")
    index.upsert_index(orphan, content="orphan")
    index.upsert_index(deleted, content="deleted")

    report = ConsistencyVerifier(source, index).verify()
    assert source_obj.uri in report.missing_index
    assert orphan.uri in report.orphan_index
    assert deleted.uri in report.deleted_in_default_search

