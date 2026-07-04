from __future__ import annotations

from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore, InMemoryRelationStore


def test_context_db_facade_reads_writes_searches_relations_and_rebuilds(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path / "source")
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()
    db = ContextDB(source, index, relations)
    obj = ContextObject(
        uri="memoryos://user/u1/memories/profile/temp",
        context_type=ContextType.MEMORY,
        title="temperature preference",
        owner_user_id="u1",
    )

    db.write_object(obj, content="prefers 26C")
    assert db.read_object(obj.uri).title == "temperature preference"
    assert [hit.uri for hit in db.search("26C", owner_user_id="u1", context_type=ContextType.MEMORY)] == [obj.uri]

    relation = ContextRelation(
        source_uri="memoryos://user/u1/action_policies/hot/turn_on_ac",
        relation_type="anchored_by",
        target_uri=obj.uri,
        metadata={"owner_user_id": "u1"},
    )
    db.add_relation(relation)
    assert db.relations_of(relation.source_uri, owner_user_id="u1")[0].target_uri == obj.uri

    index.clear()
    missing = db.verify_consistency(owner_user_id="u1")
    assert missing["source_count"] == 1
    assert missing["indexed_count"] == 0
    assert missing["missing_index"] == [obj.uri]
    assert missing["dangling_index"] == []

    rebuilt = db.rebuild_index(owner_user_id="u1")
    assert rebuilt["indexed_count"] == 1
    assert rebuilt["missing_index"] == []

    orphan = ContextObject(
        uri="memoryos://user/u1/memories/profile/orphan",
        context_type=ContextType.MEMORY,
        title="orphan",
        owner_user_id="u1",
    )
    index.upsert_index(orphan, content="orphan")
    assert orphan.uri in db.verify_consistency(owner_user_id="u1")["dangling_index"]
