from __future__ import annotations

import pytest

from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.model.context_layer import ContextLayers
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore, InMemoryRelationStore
from memoryos.operations.commit.operation_committer import OperationCommitter


def test_context_db_facade_reads_writes_searches_relations_and_rebuilds(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path / "source")
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()
    db = ContextDB(
        source,
        index,
        relations,
        committer=OperationCommitter(
            source,
            index,
            str(source.root),
            relation_store=relations,
        ),
    )
    obj = ContextObject(
        uri="memoryos://user/u1/resources/profile/temp",
        context_type=ContextType.RESOURCE,
        title="temperature preference",
        owner_user_id="u1",
        layers=ContextLayers(l2_uri="memoryos://user/u1/resources/profile/temp/body.md"),
    )

    db.write_object(obj, content="prefers 26C")
    assert db.read_object(obj.uri).title == "temperature preference"
    assert [hit.uri for hit in db.search("26C", owner_user_id="u1", context_type=ContextType.RESOURCE)] == [obj.uri]

    relation = ContextRelation(
        source_uri="memoryos://user/u1/action_policies/hot/turn_on_ac",
        relation_type="anchored_by",
        target_uri=obj.uri,
        metadata={"owner_user_id": "u1"},
    )
    db.add_relation(relation)
    assert db.relations_of(relation.source_uri, owner_user_id="u1")[0].target_uri == obj.uri
    stored = source.read_object(obj.uri)
    assert [(item.source_uri, item.relation_type, item.target_uri) for item in stored.relations] == [
        (relation.source_uri, relation.relation_type, relation.target_uri)
    ]
    assert source.read_content(stored.layers.l2_uri or stored.uri) == "prefers 26C"
    # A retry carries a new default timestamp but remains one Source fact and
    # repairs (rather than duplicates) the disposable relation row.
    db.add_relation(
        ContextRelation(
            source_uri=relation.source_uri,
            relation_type=relation.relation_type,
            target_uri=relation.target_uri,
            metadata={"owner_user_id": "u1"},
        )
    )
    assert len(source.read_object(obj.uri).relations) == 1
    assert len(db.relations_of(relation.source_uri, owner_user_id="u1")) == 1

    index.clear(tenant_id="default")
    missing = db.verify_consistency(owner_user_id="u1")
    assert missing["source_count"] == 1
    assert missing["indexed_count"] == 0
    assert missing["missing_index"] == [obj.uri]
    assert missing["dangling_index"] == []

    rebuilt = db.rebuild_index(owner_user_id="u1")
    assert rebuilt["indexed_count"] == 1
    assert rebuilt["missing_index"] == []

    orphan = ContextObject(
        uri="memoryos://user/u1/resources/profile/orphan",
        context_type=ContextType.RESOURCE,
        title="orphan",
        owner_user_id="u1",
    )
    index.upsert_index(orphan, content="orphan", tenant_id="default")
    assert orphan.uri in db.verify_consistency(owner_user_id="u1")["dangling_index"]


def test_context_db_cannot_seed_markdown_document_source(tmp_path) -> None:  # noqa: ANN001
    source = FileSystemSourceStore(tmp_path / "source", tenant_id="t1")
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()
    db = ContextDB(
        source,
        index,
        relations,
        committer=OperationCommitter(
            source,
            index,
            str(source.root),
            relation_store=relations,
            tenant_id="t1",
        ),
    )
    document = ContextObject(
        uri="memoryos://user/u1/memory/documents/memdoc_01J00000000000000000000000",
        context_type=ContextType.RESOURCE,
        title="document projection",
        owner_user_id="u1",
        tenant_id="t1",
    )

    with pytest.raises(PermissionError, match="domain-owned context cannot be seeded"):
        db.seed_object(document, content="must not become Source authority")
    with pytest.raises(PermissionError, match="not ordinary SourceStore objects"):
        source.read_object(document.uri)
