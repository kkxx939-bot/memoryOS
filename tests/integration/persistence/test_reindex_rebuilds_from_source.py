from __future__ import annotations

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore
from memoryos.contextdb.transaction.consistency import ConsistencyVerifier
from memoryos.workers.reindex_worker import ReindexWorker


def test_reindex_worker_rebuilds_index_from_source(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    obj = ContextObject(uri="memoryos://user/u1/resources/profile/a", context_type=ContextType.RESOURCE, title="alpha", owner_user_id="u1")
    source.write_object(obj, content="temperature preference")
    assert obj.uri in ConsistencyVerifier(source, index).verify().missing_index

    ReindexWorker(source, index).rebuild()
    assert not ConsistencyVerifier(source, index).verify().missing_index
    assert [
        hit.uri
        for hit in index.search(
            "temperature",
            tenant_id="default",
            filters={"owner_user_id": "u1"},
        )
    ] == [obj.uri]
