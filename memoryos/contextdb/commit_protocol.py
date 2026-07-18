"""Commit boundary owned by the generic ContextDB facade.

ContextDB accepts an implementation from the composition root.  The protocol
deliberately describes only the capability ContextDB consumes; construction of
operation-domain models remains the implementation's responsibility.
"""

from __future__ import annotations

from typing import Any, Protocol

from memoryos.contextdb.model.context_object import ContextObject
from memoryos.contextdb.store.index_store import IndexStore
from memoryos.contextdb.store.relation_store import RelationStore
from memoryos.contextdb.store.source_store import SourceStore


class ContextCommitOperation(Protocol):
    """Minimum shape accepted by the historical commit-operation facade."""

    user_id: str


class ContextCommitter(Protocol):
    """Narrow commit capability injected into ContextDB."""

    source_store: SourceStore
    index_store: IndexStore
    relation_store: RelationStore | None
    tenant_id: str
    tombstone_service: Any

    def commit(self, user_id: str, operations: list[Any]) -> Any: ...

    def commit_ordinary_relation_update(
        self,
        *,
        owner_user_id: str,
        desired_authority: ContextObject,
        content: str,
        tenant_id: str,
    ) -> Any: ...


__all__ = ["ContextCommitOperation", "ContextCommitter"]
