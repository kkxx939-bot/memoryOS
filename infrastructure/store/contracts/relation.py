"""上下文关系的持久化协议。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from infrastructure.store.model.context.context_relation import ContextRelation


class RelationStore(Protocol):
    def add_relation(self, relation: ContextRelation, *, tenant_id: str) -> None: ...

    def relations_of(
        self,
        uri: str,
        *,
        tenant_id: str,
        owner_user_id: str | None = None,
        limit: int | None = None,
    ) -> list[ContextRelation]: ...

    def delete_relation(
        self,
        source_uri: str,
        relation_type: str,
        target_uri: str,
        *,
        tenant_id: str,
    ) -> None: ...

    def delete_projection_relations(
        self,
        uri: str,
        *,
        tenant_id: str,
        catalog_record_key: str,
        limit: int,
    ) -> int: ...

    def delete_memory_document_relations(
        self,
        uri: str,
        *,
        tenant_id: str,
        owner_user_id: str,
        limit: int,
    ) -> int: ...

    def delete_uri_relations(self, uri: str, *, tenant_id: str, limit: int) -> int: ...

    def clear_ordinary_relations(self, *, tenant_id: str, limit: int) -> int: ...

    def reconcile_ordinary_relations(
        self,
        relations: Sequence[ContextRelation],
        *,
        tenant_id: str,
    ) -> dict[str, int]: ...

    def all_relations(self, *, tenant_id: str) -> list[ContextRelation]: ...


__all__ = ["RelationStore"]
