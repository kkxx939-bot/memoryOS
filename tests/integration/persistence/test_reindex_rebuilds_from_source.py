from __future__ import annotations

from infrastructure.context.maintenance.index_consistency import IndexConsistencyService
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_type import ContextType
from tests.support.persistence import FileSystemSourceStore, InMemoryIndexStore


def test_index_consistency_service_rebuilds_index_from_source(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    obj = ContextObject(uri="memoryos://user/u1/resources/profile/a", context_type=ContextType.RESOURCE, title="alpha", owner_user_id="u1")
    source.write_object(obj, content="temperature preference")
    service = IndexConsistencyService(source, index, tenant_id="default")
    assert obj.uri in service.verify().missing_in_index

    service.rebuild()
    assert not service.verify().missing_in_index
    assert [
        hit.uri
        for hit in index.search(
            "temperature",
            tenant_id="default",
            filters={"owner_user_id": "u1"},
        )
    ] == [obj.uri]
