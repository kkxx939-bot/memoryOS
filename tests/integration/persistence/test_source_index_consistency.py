from __future__ import annotations

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.index_consistency import IndexConsistencyService
from memoryos.contextdb.store.local_stores import FileSystemSourceStore, InMemoryIndexStore
from memoryos.contextdb.store.source_store import IndexHit
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore
from memoryos.contextdb.store.sqlite_relation_store import SQLiteRelationStore
from memoryos.contextdb.transaction.consistency import ConsistencyVerifier


def test_consistency_reports_missing_orphan_and_deleted_hot_index(tmp_path) -> None:
    class BrokenLifecycleIndex(InMemoryIndexStore):
        def search(
            self,
            query: str,
            *,
            tenant_id: str,
            filters: dict | None = None,
            limit: int = 10,
        ) -> list[IndexHit]:
            filters = filters or {}
            hits = []
            for (row_tenant, _uri), (obj, content) in self.rows.items():
                if row_tenant != tenant_id:
                    continue
                if filters.get("owner_user_id") and obj.owner_user_id != filters["owner_user_id"]:
                    continue
                if filters.get("context_type") and obj.context_type.value != filters["context_type"]:
                    continue
                if str(query).lower() not in f"{obj.title} {content}".lower():
                    continue
                hits.append(IndexHit(uri=obj.uri, score=1.0, context_type=obj.context_type.value, title=obj.title))
            return hits[:limit]

    source = FileSystemSourceStore(tmp_path)
    index = BrokenLifecycleIndex()
    source_obj = ContextObject(uri="memoryos://user/u1/resources/profile/a", context_type=ContextType.RESOURCE, title="a", owner_user_id="u1")
    orphan = ContextObject(uri="memoryos://user/u1/resources/profile/orphan", context_type=ContextType.RESOURCE, title="orphan", owner_user_id="u1")
    deleted = ContextObject(uri="memoryos://user/u1/resources/profile/deleted", context_type=ContextType.RESOURCE, title="deleted", owner_user_id="u1", lifecycle_state=LifecycleState.DELETED)
    source.write_object(source_obj, content="alpha")
    source.write_object(deleted, content="deleted")
    index.upsert_index(orphan, content="orphan", tenant_id="default")
    index.upsert_index(deleted, content="deleted", tenant_id="default")

    report = ConsistencyVerifier(source, index).verify()
    assert source_obj.uri in report.missing_index
    assert orphan.uri in report.orphan_index
    assert deleted.uri in report.deleted_in_default_search


def test_index_consistency_service_supports_sqlite_index_and_relations(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path / "source")
    index = SQLiteIndexStore(tmp_path / "index.sqlite3")
    relations = SQLiteRelationStore(tmp_path / "relations.sqlite3")
    source_obj = ContextObject(
        uri="memoryos://user/u1/resources/profile/a",
        context_type=ContextType.RESOURCE,
        title="alpha",
        owner_user_id="u1",
    )
    archived = ContextObject(
        uri="memoryos://user/u1/resources/profile/archived",
        context_type=ContextType.RESOURCE,
        title="archived alpha",
        owner_user_id="u1",
        lifecycle_state=LifecycleState.ARCHIVED,
    )
    orphan = ContextObject(
        uri="memoryos://user/u1/resources/profile/orphan",
        context_type=ContextType.RESOURCE,
        title="orphan alpha",
        owner_user_id="u1",
    )
    source.write_object(source_obj, content="alpha source")
    source.write_object(archived, content="archived alpha")
    index.upsert_index(orphan, content="orphan alpha", tenant_id="default")
    index.upsert_index(archived, content="archived alpha", tenant_id="default")
    relations.add_relation(
        ContextRelation(
            source_uri=source_obj.uri,
            relation_type="evidence_for",
            target_uri="memoryos://user/u1/behavior/cases/missing",
            metadata={"owner_user_id": "u1"},
        ),
        tenant_id="default",
    )

    report = IndexConsistencyService(
        source,
        index,
        relations,
        tenant_id="default",
    ).verify()
    assert source_obj.uri in report.missing_in_index
    assert orphan.uri in report.orphan_index
    assert archived.uri not in report.deleted_or_archived_in_default_search
    assert report.broken_relations

    rebuilt = IndexConsistencyService(source, index, tenant_id="default").rebuild()
    assert not rebuilt.missing_in_index
    assert not rebuilt.orphan_index
    assert not index.search(
        "archived",
        tenant_id="default",
        filters={"owner_user_id": "u1"},
    )
