"""Optional domain overlays consumed by the generic ContextDB facade.

ContextDB owns only the extension protocol.  A domain that needs committed
views or protected write semantics provides the implementation at the
composition root; ContextDB never imports that domain.
"""

from __future__ import annotations

from typing import Any, Protocol

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.model.context_relation import ContextRelation
from memoryos.contextdb.store.relation_store import RelationStore
from memoryos.contextdb.store.source_store import SourceStore


class ContextDomainClassifier(Protocol):
    """Classify objects owned by an optional domain extension."""

    def owns_uri(self, uri: str) -> bool: ...

    def owns_object(self, obj: ContextObject) -> bool: ...


class ContextDomainOverlay(ContextDomainClassifier, Protocol):
    """Narrow extension point for domain-owned committed context views."""

    def owns_uri(self, uri: str) -> bool: ...

    def owns_object(self, obj: ContextObject) -> bool: ...

    def read_object(
        self,
        source_store: SourceStore,
        relation_store: RelationStore,
        uri: str,
    ) -> ContextObject: ...

    def relations_of(
        self,
        source_store: SourceStore,
        relation_store: RelationStore,
        uri: str,
        *,
        owner_user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> list[ContextRelation]: ...


class ContextIndexPolicy(Protocol):
    """Protect domain-owned serving rows during a generic index rebuild."""

    def owns_index_entry(
        self,
        source_store: SourceStore,
        uri: str,
        metadata: dict[str, Any] | None,
    ) -> bool: ...

    def preserve_index_entry(
        self,
        source_store: SourceStore,
        index_store: Any,
        uri: str,
        metadata: dict[str, Any] | None,
    ) -> bool: ...


class NoDomainOverlay:
    """Default overlay for a standalone, domain-neutral ContextDB."""

    def owns_uri(self, uri: str) -> bool:
        del uri
        return False

    def owns_object(self, obj: ContextObject) -> bool:
        del obj
        return False

    def read_object(
        self,
        source_store: SourceStore,
        relation_store: RelationStore,
        uri: str,
    ) -> ContextObject:
        del relation_store
        return source_store.read_object(uri)

    def relations_of(
        self,
        source_store: SourceStore,
        relation_store: RelationStore,
        uri: str,
        *,
        owner_user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> list[ContextRelation]:
        del source_store
        return relation_store.relations_of(
            uri,
            owner_user_id=owner_user_id,
            tenant_id=tenant_id,
        )


class NoContextIndexPolicy:
    """Default policy for a domain-neutral ContextDB."""

    def owns_index_entry(
        self,
        source_store: SourceStore,
        uri: str,
        metadata: dict[str, Any] | None,
    ) -> bool:
        del source_store, uri, metadata
        return False

    def preserve_index_entry(
        self,
        source_store: SourceStore,
        index_store: Any,
        uri: str,
        metadata: dict[str, Any] | None,
    ) -> bool:
        del source_store, index_store, uri, metadata
        return False


__all__ = [
    "ContextDomainClassifier",
    "ContextDomainOverlay",
    "ContextIndexPolicy",
    "NoContextIndexPolicy",
    "NoDomainOverlay",
]
