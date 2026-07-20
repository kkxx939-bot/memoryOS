"""索引和 Catalog 存储需要实现的协议。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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

    def delete_catalog(self, record_key: str, *, tenant_id: str) -> bool: ...


class MemoryDocumentProjectionStore(Protocol):
    """单个 Markdown 文档的原子 Serving 发布协议。"""

    def get_memory_document_projection_state(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
    ) -> Mapping[str, object] | None: ...

    def replace_memory_document_projection(
        self,
        document_record: CatalogRecord | Mapping[str, object],
        block_records: Sequence[CatalogRecord | Mapping[str, object]],
        expected_previous_generation: int | None,
        *,
        tenant_id: str,
        owner_user_id: str,
        restore_soft_deleted: bool = False,
    ) -> tuple[str, ...]: ...

    def tombstone_memory_document_projection(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        document_id: str,
        deletion_generation: int,
        deletion_event_digest: str,
        deletion_status: str,
        relative_path: str = "",
    ) -> tuple[str, ...]: ...


__all__ = ["CatalogStore", "IndexHit", "IndexStore", "MemoryDocumentProjectionStore"]
