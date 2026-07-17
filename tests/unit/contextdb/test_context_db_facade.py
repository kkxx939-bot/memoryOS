from __future__ import annotations

import pytest

from memoryos.contextdb.context_db import ContextDB
from memoryos.contextdb.model.context_layer import ContextLayers
from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore, InMemoryRelationStore
from memoryos.memory.integration.context_overlay import CanonicalMemoryContextOverlay
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
        uri="memoryos://user/u1/memories/profile/temp",
        context_type=ContextType.MEMORY,
        title="temperature preference",
        owner_user_id="u1",
        layers=ContextLayers(l2_uri="memoryos://user/u1/memories/profile/temp/body.md"),
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


def test_context_db_add_relation_rejects_unproved_canonical_target(tmp_path) -> None:  # noqa: ANN001
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
    db._configure_extensions(domain_overlay=CanonicalMemoryContextOverlay())
    policy = ContextObject(
        uri="memoryos://user/u1/action_policies/unproved-canonical",
        context_type=ContextType.ACTION_POLICY,
        title="unproved canonical relation source",
        owner_user_id="u1",
        tenant_id="t1",
    )
    claim = ContextObject(
        uri="memoryos://user/u1/memories/canonical/slots/s1/claims/c1",
        context_type=ContextType.MEMORY,
        title="unproved claim",
        owner_user_id="u1",
        tenant_id="t1",
        metadata={"canonical_kind": "claim", "claim_id": "c1", "state": "ACTIVE"},
        schema_version="canonical_memory_v2",
    )
    db.seed_object(policy, content="policy")
    source.write_object(claim, content="unproved")
    edge = ContextRelation(
        source_uri=policy.uri,
        relation_type="constrained_by",
        target_uri=claim.uri,
        metadata={"tenant_id": "t1", "owner_user_id": "u1"},
    )

    with pytest.raises(FileNotFoundError, match="no committed transaction proof"):
        db.add_relation(edge)
    assert not source.read_object(policy.uri).relations
    assert not relations.relations_of(policy.uri, tenant_id="t1")
