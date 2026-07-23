"""索引和 Catalog 存储需要实现的协议。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from infrastructure.store.model.catalog import CatalogRecord
from infrastructure.store.model.context.context_object import ContextObject


@dataclass(frozen=True)
class IndexHit:
    uri: str
    score: float
    context_type: str
    title: str = ""
    layer: str = "l0"
    metadata: dict = field(default_factory=dict)


class IndexStore(Protocol):
    def upsert_index(self, obj: ContextObject, content: str = "", *, tenant_id: str) -> None: ...

    def delete_index(self, uri: str, *, tenant_id: str) -> None: ...

    def indexed_uris(self, *, tenant_id: str) -> list[str]: ...

    def clear(self, *, tenant_id: str) -> None: ...

    def search(
        self,
        query: str,
        *,
        tenant_id: str,
        filters: dict | None = None,
        limit: int = 10,
    ) -> list[IndexHit]: ...

    def get_index_metadata(self, uri: str, *, tenant_id: str) -> dict | None: ...

    def ordinary_relation_endpoint_state(
        self,
        uri: str,
        *,
        tenant_id: str,
        session_id: str = "",
    ) -> str: ...


class CatalogStore(Protocol):
    """显式限定租户的 Catalog Serving 记录精确读写协议。"""

    def upsert_catalog(
        self,
        record: CatalogRecord | Mapping[str, object],
        *,
        tenant_id: str,
    ) -> None: ...

    def get_catalog(self, record_key: str, *, tenant_id: str) -> CatalogRecord | None: ...

    def get_catalog_by_uri(
        self,
        uri: str,
        *,
        tenant_id: str,
        limit: int = 100,
    ) -> list[CatalogRecord]: ...

    def delete_catalog(self, record_key: str, *, tenant_id: str) -> bool: ...


__all__ = ["CatalogStore", "IndexHit", "IndexStore"]
