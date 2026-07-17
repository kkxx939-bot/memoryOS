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
        def search(self, query: str, filters: dict | None = None, limit: int = 10) -> list[IndexHit]:
            filters = filters or {}
            hits = []
            for obj, content in self.rows.values():
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
    source_obj = ContextObject(uri="memoryos://user/u1/memories/profile/a", context_type=ContextType.MEMORY, title="a", owner_user_id="u1")
    orphan = ContextObject(uri="memoryos://user/u1/memories/profile/orphan", context_type=ContextType.MEMORY, title="orphan", owner_user_id="u1")
    deleted = ContextObject(uri="memoryos://user/u1/memories/profile/deleted", context_type=ContextType.MEMORY, title="deleted", owner_user_id="u1", lifecycle_state=LifecycleState.DELETED)
    source.write_object(source_obj, content="alpha")
    source.write_object(deleted, content="deleted")
    index.upsert_index(orphan, content="orphan")
    index.upsert_index(deleted, content="deleted")

    report = ConsistencyVerifier(source, index).verify()
    assert source_obj.uri in report.missing_index
    assert orphan.uri in report.orphan_index
    assert deleted.uri in report.deleted_in_default_search


def test_index_consistency_service_supports_sqlite_index_and_relations(tmp_path) -> None:
    source = FileSystemSourceStore(tmp_path / "source")
    index = SQLiteIndexStore(tmp_path / "index.sqlite3")
    relations = SQLiteRelationStore(tmp_path / "relations.sqlite3")
    source_obj = ContextObject(
        uri="memoryos://user/u1/memories/profile/a",
        context_type=ContextType.MEMORY,
        title="alpha",
        owner_user_id="u1",
    )
    archived = ContextObject(
        uri="memoryos://user/u1/memories/profile/archived",
        context_type=ContextType.MEMORY,
        title="archived alpha",
        owner_user_id="u1",
        lifecycle_state=LifecycleState.ARCHIVED,
    )
    orphan = ContextObject(
        uri="memoryos://user/u1/memories/profile/orphan",
        context_type=ContextType.MEMORY,
        title="orphan alpha",
        owner_user_id="u1",
    )
    source.write_object(source_obj, content="alpha source")
    source.write_object(archived, content="archived alpha")
    index.upsert_index(orphan, content="orphan alpha")
    index.upsert_index(archived, content="archived alpha")
    relations.add_relation(
        ContextRelation(
            source_uri=source_obj.uri,
            relation_type="evidence_for",
            target_uri="memoryos://user/u1/behavior/cases/missing",
            metadata={"owner_user_id": "u1"},
        )
    )

    report = IndexConsistencyService(source, index, relations).verify()
    assert source_obj.uri in report.missing_in_index
    assert orphan.uri in report.orphan_index
    assert archived.uri not in report.deleted_or_archived_in_default_search
    assert report.broken_relations

    rebuilt = IndexConsistencyService(source, index).rebuild()
    assert not rebuilt.missing_in_index
    assert not rebuilt.orphan_index
    assert not index.search("archived", filters={"owner_user_id": "u1"})
