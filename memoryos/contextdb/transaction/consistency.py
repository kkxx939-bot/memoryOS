"""上下文数据库里的一致性检查。"""

from __future__ import annotations

from dataclasses import dataclass, field

from memoryos.contextdb.model.lifecycle import LifecycleState
from memoryos.contextdb.store.source_store import (
    IndexStore,
    RelationStore,
    SourceStore,
    is_canonical_memory_object,
    is_canonical_memory_uri,
)


@dataclass(frozen=True)
class ConsistencyReport:
    missing_index: list[str] = field(default_factory=list)
    orphan_index: list[str] = field(default_factory=list)
    deleted_in_default_search: list[str] = field(default_factory=list)
    broken_relations: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.missing_index or self.orphan_index or self.deleted_in_default_search or self.broken_relations)


class ConsistencyVerifier:
    def __init__(self, source_store: SourceStore, index_store: IndexStore, relation_store: RelationStore | None = None) -> None:
        self.source_store = source_store
        self.index_store = index_store
        self.relation_store = relation_store

    def verify(self) -> ConsistencyReport:
        tenant_id = str(getattr(self.source_store, "tenant_id", "default") or "default")
        objects = [
            obj
            for obj in self.source_store.list_objects()
            if not is_canonical_memory_object(obj)
        ]
        source_uris = {obj.uri for obj in objects}
        missing_index = []
        deleted_in_default_search = []
        inactive_states = {LifecycleState.DELETED, LifecycleState.ARCHIVED, LifecycleState.OBSOLETE}
        for obj in objects:
            hits = self.index_store.search(
                obj.title or obj.uri,
                filters={
                    "tenant_id": tenant_id,
                    "context_type": obj.context_type.value,
                    "owner_user_id": obj.owner_user_id,
                },
                limit=20,
            )
            hit_uris = {hit.uri for hit in hits}
            if obj.lifecycle_state not in inactive_states and obj.uri not in hit_uris:
                missing_index.append(obj.uri)
            if obj.lifecycle_state in inactive_states and obj.uri in hit_uris:
                deleted_in_default_search.append(obj.uri)
        indexed_uris = {
            uri
            for uri in getattr(self.index_store, "indexed_uris", lambda: [])()
            if not self._canonical_uri(uri)
        }
        orphan_index = sorted(uri for uri in indexed_uris if uri not in source_uris)
        broken_relations = self._broken_relations(source_uris, tenant_id=tenant_id)
        return ConsistencyReport(
            missing_index=sorted(missing_index),
            orphan_index=orphan_index,
            deleted_in_default_search=sorted(deleted_in_default_search),
            broken_relations=broken_relations,
        )

    def _broken_relations(self, source_uris: set[str], *, tenant_id: str) -> list[dict]:
        if self.relation_store is None:
            return []
        broken = []
        for uri in source_uris:
            for relation in self.relation_store.relations_of(uri, tenant_id=tenant_id):
                if self._global_uri(relation.source_uri) or self._global_uri(relation.target_uri):
                    continue
                if self._canonical_uri(relation.source_uri) or self._canonical_uri(relation.target_uri):
                    continue
                if relation.source_uri not in source_uris or relation.target_uri not in source_uris:
                    broken.append(relation.to_dict())
        return broken

    def _global_uri(self, uri: str) -> bool:
        return uri.startswith(("memoryos://resources/", "memoryos://skills/"))

    @staticmethod
    def _canonical_uri(uri: str) -> bool:
        return is_canonical_memory_uri(uri)
