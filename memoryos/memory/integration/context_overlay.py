"""Canonical-memory implementation of ContextDB's domain overlay protocol."""

from __future__ import annotations

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.store.relation_store import RelationStore
from memoryos.contextdb.store.source_store import SourceStore
from memoryos.memory.canonical.visibility import (
    list_committed_relations,
    read_committed_canonical,
)
from memoryos.memory.integration.classification import (
    is_canonical_memory_object,
    is_canonical_memory_uri,
)


class CanonicalMemoryContextOverlay:
    """Expose receipt-committed canonical state through generic ContextDB."""

    def owns_uri(self, uri: str) -> bool:
        return is_canonical_memory_uri(uri)

    def owns_object(self, obj: ContextObject) -> bool:
        return is_canonical_memory_object(obj)

    def read_object(
        self,
        source_store: SourceStore,
        relation_store: RelationStore,
        uri: str,
    ) -> ContextObject:
        return read_committed_canonical(source_store, uri, relation_store).object

    def relations_of(
        self,
        source_store: SourceStore,
        relation_store: RelationStore,
        uri: str,
        *,
        owner_user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> list[ContextRelation]:
        read_committed_canonical(source_store, uri, relation_store)
        relations = list(list_committed_relations(source_store, uri, relation_store))
        if tenant_id is not None:
            relations = [
                relation
                for relation in relations
                if str(relation.metadata.get("tenant_id") or "default") == tenant_id
            ]
        if owner_user_id is not None:
            relations = [
                relation
                for relation in relations
                if str(relation.metadata.get("owner_user_id") or "") == owner_user_id
            ]
        return relations


__all__ = ["CanonicalMemoryContextOverlay"]
