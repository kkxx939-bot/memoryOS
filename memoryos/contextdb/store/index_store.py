"""ContextDB index protocol and result model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from memoryos.contextdb.model.context_object import ContextObject


@dataclass(frozen=True)
class IndexHit:
    uri: str
    score: float
    context_type: str
    title: str = ""
    layer: str = "l0"
    metadata: dict = field(default_factory=dict)


class IndexStore(Protocol):
    def upsert_index(self, obj: ContextObject, content: str = "") -> None: ...

    def delete_index(self, uri: str) -> None: ...

    def indexed_uris(self) -> list[str]: ...

    def clear(self) -> None: ...

    def search(self, query: str, filters: dict | None = None, limit: int = 10) -> list[IndexHit]: ...

    def get_index_metadata(self, uri: str) -> dict | None: ...

    def ordinary_relation_endpoint_state(
        self,
        uri: str,
        *,
        tenant_id: str,
        session_id: str = "",
    ) -> str: ...


__all__ = ["IndexHit", "IndexStore"]
