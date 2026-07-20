from __future__ import annotations

from infrastructure.context.facade import ContextDB
from infrastructure.context.maintenance import GenericContextMaintenance
from infrastructure.context.operation_effects import InfrastructureContextOperationEffects
from infrastructure.store.model.context.context_layer import ContextLayers
from infrastructure.store.model.context.context_object import ContextObject
from infrastructure.store.model.context.context_relation import ContextRelation
from infrastructure.store.model.context.context_type import ContextType
from tests.support.persistence import (
    FileSystemSourceStore,
    InMemoryIndexStore,
    InMemoryRelationStore,
    seed_context_object,
)
from tests.support.transaction import build_test_operation_committer as OperationCommitter


def test_context_db_facade_reads_searches_and_coordinates_relations(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path / "source")
    index = InMemoryIndexStore()
    relations = InMemoryRelationStore()
    db = ContextDB(
        source,
        index,
        relations,
        relation_committer=OperationCommitter(
            source,
            index,
            str(source.root),
            relation_store=relations,
            context_effects=InfrastructureContextOperationEffects(),
        ),
    )
    administration = GenericContextMaintenance(
        source,
        index,
        relations,
        tenant_id="default",
        serving_lock=db.serving_lock,
    )
    obj = ContextObject(
        uri="memoryos://user/u1/resources/profile/temp",
        context_type=ContextType.RESOURCE,
        title="temperature preference",
        owner_user_id="u1",
        layers=ContextLayers(l2_uri="memoryos://user/u1/resources/profile/temp/body.md"),
    )

    seed_context_object(source, index, obj, content="prefers 26C")
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
    missing = administration.verify_consistency(owner_user_id="u1")
    assert missing["source_count"] == 1
    assert missing["indexed_count"] == 0
    assert missing["missing_index"] == [obj.uri]
    assert missing["dangling_index"] == []

    rebuilt = administration.rebuild_index(owner_user_id="u1")
    assert rebuilt["indexed_count"] == 1
    assert rebuilt["missing_index"] == []

    orphan = ContextObject(
        uri="memoryos://user/u1/resources/profile/orphan",
        context_type=ContextType.RESOURCE,
        title="orphan",
        owner_user_id="u1",
    )
    index.upsert_index(orphan, content="orphan", tenant_id="default")
    assert orphan.uri in administration.verify_consistency(owner_user_id="u1")["dangling_index"]


def test_add_relation_requires_explicit_commit_capability(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path)
    index = InMemoryIndexStore()
    db = ContextDB(source, index, InMemoryRelationStore())
    obj = ContextObject(
        uri="memoryos://user/u1/resources/profile/relation-target",
        context_type=ContextType.RESOURCE,
        title="relation target",
        owner_user_id="u1",
    )
    seed_context_object(source, index, obj, content="relation target")

    try:
        db.add_relation(
            ContextRelation(
                source_uri="memoryos://user/u1/resources/source",
                relation_type="related_to",
                target_uri=obj.uri,
                metadata={"owner_user_id": "u1"},
            )
        )
    except RuntimeError as exc:
        assert "injected OrdinaryRelationCommitter" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("add_relation should require an injected commit capability")
    assert db.relation_committer is None
