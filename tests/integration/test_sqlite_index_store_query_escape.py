from __future__ import annotations

import math

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


def test_chinese_ngram_fallback_matches_non_contiguous_phrase(tmp_path) -> None:
    store = SQLiteIndexStore(tmp_path / "index.sqlite3")
    obj = ContextObject(
        uri="memoryos://user/u1/memories/m1",
        context_type=ContextType.MEMORY,
        title="空调偏好",
        owner_user_id="u1",
    )
    store.upsert_index(obj, content="用户喜欢热的时候开空调")

    hits = store.search("喜欢开空调", filters={"owner_user_id": "u1"}, limit=5)

    assert [hit.uri for hit in hits] == [obj.uri]
    assert hits[0].metadata["retrieval_scores"]["lexical"] == 0.75


def test_zero_relevance_is_not_promoted_by_hotness(tmp_path) -> None:
    store = SQLiteIndexStore(tmp_path / "index.sqlite3")
    unrelated = ContextObject(
        uri="memoryos://user/u1/memories/unrelated",
        context_type=ContextType.MEMORY,
        title="天气偏好",
        owner_user_id="u1",
        hotness=1.0,
        semantic_hotness=1.0,
        behavior_support_hotness=1.0,
    )
    store.upsert_index(unrelated, content="晴天适合户外活动")

    assert store.search("PostgreSQL", filters={"owner_user_id": "u1"}) == []


def test_contains_fallback_keeps_hotness_behind_base_relevance(tmp_path) -> None:
    store = SQLiteIndexStore(tmp_path / "index.sqlite3")
    relevant = ContextObject(
        uri="memoryos://user/u1/memories/relevant",
        context_type=ContextType.MEMORY,
        title="database",
        owner_user_id="u1",
    )
    unrelated = ContextObject(
        uri="memoryos://user/u1/memories/hot",
        context_type=ContextType.MEMORY,
        title="weather",
        owner_user_id="u1",
        hotness=1.0,
        semantic_hotness=1.0,
        behavior_support_hotness=1.0,
    )
    store.upsert_index(relevant, content="PostgreSQL database")
    store.upsert_index(unrelated, content="sunny outdoor activity")
    store.fts_enabled = False

    hits = store.search("PostgreSQL", filters={"owner_user_id": "u1"})

    assert [hit.uri for hit in hits] == [relevant.uri]
    scores = hits[0].metadata["retrieval_scores"]
    assert scores == {
        "lexical": 1.0,
        "vector": 0.0,
        "identity": 0.0,
        "base_relevance": 1.0,
        "hotness": 0.0,
        "score": 1.0,
    }


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
    assert hits[0].metadata["retrieval_scores"]["identity"] == 1.0
    assert hits[0].metadata["retrieval_scores"]["base_relevance"] == 1.0


def test_nan_rank_and_hotness_cannot_create_non_finite_or_zero_base_hit(tmp_path) -> None:
    store = SQLiteIndexStore(tmp_path / "index.sqlite3")
    obj = ContextObject(
        uri="memoryos://user/u1/memories/m1",
        context_type=ContextType.MEMORY,
        title="database",
        owner_user_id="u1",
        hotness=1.0,
        semantic_hotness=1.0,
        behavior_support_hotness=1.0,
    )
    store.upsert_index(obj, content="PostgreSQL database")
    with store._connect() as conn:  # noqa: SLF001
        conn.execute("UPDATE contexts SET hotness = ? WHERE uri = ?", ("NaN", obj.uri))
        row = conn.execute("SELECT * FROM contexts WHERE uri = ?", (obj.uri,)).fetchone()
    scores = store._score_components(row, lexical=1.0, lexical_rank=float("nan"))  # noqa: SLF001
    zero = store._score_components(row, lexical=0.0, lexical_rank=float("nan"))  # noqa: SLF001
    boolean = store._score_components(  # noqa: SLF001
        row,
        lexical=True,
        lexical_rank=True,
        vector=True,
        identity=True,
        identity_rank=True,
    )

    assert math.isfinite(scores["score"])
    assert math.isfinite(scores["hotness"])
    assert scores["base_relevance"] == 1.0
    assert zero["base_relevance"] == 0.0
    assert zero["score"] == 0.0
    assert boolean["base_relevance"] == 0.0
    assert boolean["score"] == 0.0
