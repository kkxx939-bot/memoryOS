from __future__ import annotations

from dataclasses import dataclass, field

from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.source_store import IndexStore, RelationStore, SourceStore


@dataclass(frozen=True)
class IndexConsistencyResult:
    source_count: int
    index_count: int
    missing_in_index: list[str]
    orphan_index: list[str] = field(default_factory=list)
    deleted_or_archived_in_default_search: list[str] = field(default_factory=list)
    broken_relations: list[dict] = field(default_factory=list)

    @property
    def consistent(self) -> bool:
        return not (
            self.missing_in_index
            or self.orphan_index
            or self.deleted_or_archived_in_default_search
            or self.broken_relations
        )


class IndexConsistencyService:
    def __init__(
        self,
        source_store: SourceStore,
        index_store: IndexStore,
        relation_store: RelationStore | None = None,
    ) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.relation_store = relation_store

    def verify(self) -> IndexConsistencyResult:
        objects = self.source_store.list_objects()
        source_uris = {obj.uri for obj in objects}
        indexed_uris = set(self.index_store.indexed_uris())
        missing = sorted(
            obj.uri
            for obj in objects
            if obj.lifecycle_state not in {LifecycleState.DELETED, LifecycleState.ARCHIVED}
            and obj.uri not in indexed_uris
        )
        orphan = sorted(uri for uri in indexed_uris if uri not in source_uris)
        hot_in_default = []
        for obj in objects:
            if obj.lifecycle_state not in {LifecycleState.DELETED, LifecycleState.ARCHIVED}:
                continue
            hits = self.index_store.search(
                obj.title or obj.uri,
                filters={"owner_user_id": obj.owner_user_id, "context_type": obj.context_type.value},
                limit=50,
            )
            if obj.uri in {hit.uri for hit in hits}:
                hot_in_default.append(obj.uri)
        return IndexConsistencyResult(
            source_count=len(objects),
            index_count=len(indexed_uris),
            missing_in_index=missing,
            orphan_index=orphan,
            deleted_or_archived_in_default_search=sorted(hot_in_default),
            broken_relations=self._broken_relations(source_uris),
        )

    def rebuild(self) -> IndexConsistencyResult:
        self.index_store.clear()
        for obj in self.source_store.list_objects():
            if obj.lifecycle_state in {LifecycleState.DELETED, LifecycleState.ARCHIVED}:
                self.index_store.delete_index(obj.uri)
                continue
            try:
                content = self.source_store.read_content(obj.uri)
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                content = ""
            self.index_store.upsert_index(obj, content=content)
        return self.verify()

    def _broken_relations(self, source_uris: set[str]) -> list[dict]:
        if self.relation_store is None:
            return []
        broken: list[dict] = []
        seen: set[tuple[str, str, str]] = set()
        for uri in source_uris:
            for relation in self.relation_store.relations_of(uri):
                key = (relation.source_uri, relation.relation_type, relation.target_uri)
                if key in seen:
                    continue
                seen.add(key)
                if self._global_uri(relation.source_uri) or self._global_uri(relation.target_uri):
                    continue
                if relation.source_uri not in source_uris or relation.target_uri not in source_uris:
                    broken.append(relation.to_dict())
        return broken

    def _global_uri(self, uri: str) -> bool:
        return uri.startswith(("memoryos://resources/", "memoryos://skills/"))
