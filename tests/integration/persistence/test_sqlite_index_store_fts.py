from __future__ import annotations

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore


def test_sqlite_index_store_fts_or_fallback_search_and_filters(tmp_path) -> None:
    store = SQLiteIndexStore(tmp_path / "index.sqlite3")
    u1 = ContextObject(
        uri="memoryos://user/u1/memories/profile/temp",
        context_type=ContextType.MEMORY,
        title="室温偏好",
        owner_user_id="u1",
        metadata={"summary": "用户喜欢 26 度"},
        semantic_hotness=0.4,
    )
    u2 = ContextObject(
        uri="memoryos://user/u2/memories/profile/temp",
        context_type=ContextType.MEMORY,
        title="temperature preference",
        owner_user_id="u2",
        metadata={"summary": "likes 23 degrees"},
    )
    deleted = ContextObject(
        uri="memoryos://user/u1/memories/profile/deleted",
        context_type=ContextType.MEMORY,
        title="deleted temperature",
        owner_user_id="u1",
        lifecycle_state=LifecycleState.DELETED,
    )
    archived = ContextObject(
        uri="memoryos://user/u1/memories/profile/archived",
        context_type=ContextType.MEMORY,
        title="archived temperature",
        owner_user_id="u1",
        lifecycle_state=LifecycleState.ARCHIVED,
    )
    store.upsert_index(u1, content="hot room air conditioner 温度")
    store.upsert_index(u2, content="hot room")
    store.upsert_index(deleted, content="hot room")
    store.upsert_index(archived, content="hot room")

    chinese_hits = store.search("温度", filters={"owner_user_id": "u1", "context_type": ContextType.MEMORY.value})
    assert [hit.uri for hit in chinese_hits] == [u1.uri]

    english_hits = store.search("hot", filters={"owner_user_id": "u2", "context_type": ContextType.MEMORY.value})
    assert [hit.uri for hit in english_hits] == [u2.uri]

    default_hits = store.search("hot", filters={"owner_user_id": "u1"}, limit=10)
    assert deleted.uri not in {hit.uri for hit in default_hits}
    assert archived.uri not in {hit.uri for hit in default_hits}

    store.delete_index(u1.uri)
    assert not store.search("温度", filters={"owner_user_id": "u1"})
