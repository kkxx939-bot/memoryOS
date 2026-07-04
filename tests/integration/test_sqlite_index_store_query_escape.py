from __future__ import annotations

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore


def test_fts_query_escapes_uri_punctuation(tmp_path) -> None:
    store = SQLiteIndexStore(tmp_path / "index.sqlite3")
    uri = "memoryos://user/u1/action_policies/hot_room/turn_on_ac"
    obj = ContextObject(
        uri=uri,
        context_type=ContextType.ACTION_POLICY,
        title="AC policy",
        owner_user_id="u1",
        metadata={"scene_key": "hot_room", "action": "turn_on_ac", "memory_anchor_uri": "memoryos://user/u1/memories/anchors/hot"},
    )
    store.upsert_index(obj, content="hot room turn_on_ac")

    hits = store.search("memoryos://user/u1/memories/anchors/hot", filters={"owner_user_id": "u1"}, limit=5)

    assert hits[0].uri == uri


def test_chinese_query_without_spaces_falls_back_to_contains(tmp_path) -> None:
    store = SQLiteIndexStore(tmp_path / "index.sqlite3")
    obj = ContextObject(
        uri="memoryos://user/u1/memories/m1",
        context_type=ContextType.MEMORY,
        title="偏好",
        owner_user_id="u1",
        metadata={"summary": "用户喜欢热的时候开空调"},
    )
    store.upsert_index(obj, content="用户喜欢热的时候开空调")

    hits = store.search("用户喜欢热的时候开空调", filters={"owner_user_id": "u1"}, limit=5)

    assert hits[0].uri == obj.uri


def test_metadata_exact_scene_key_and_action_are_prioritized(tmp_path) -> None:
    store = SQLiteIndexStore(tmp_path / "index.sqlite3")
    exact = ContextObject(
        uri="memoryos://user/u1/action_policies/hot_room/turn_on_ac",
        context_type=ContextType.ACTION_POLICY,
        title="Exact",
        owner_user_id="u1",
        metadata={"scene_key": "hot_room", "action": "turn_on_ac"},
    )
    lexical = ContextObject(
        uri="memoryos://user/u1/memories/noise",
        context_type=ContextType.MEMORY,
        title="hot_room",
        owner_user_id="u1",
        metadata={"summary": "hot_room appears in text"},
    )
    store.upsert_index(lexical, content="hot_room hot_room hot_room")
    store.upsert_index(exact, content="policy")

    hits = store.search("hot_room", filters={"owner_user_id": "u1"}, limit=5)

    assert hits[0].uri == exact.uri
