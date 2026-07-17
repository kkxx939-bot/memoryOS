"""ContextDB-owned vector storage protocol and serving contracts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class VectorHit:
    uri: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class VectorCapabilities:
    supports_metadata_filtering: bool = False
    supports_namespace_filtering: bool = False
    supports_time_filtering: bool = False
    supports_delete_by_filter: bool = False

    @property
    def production_filtered_top_k_ready(self) -> bool:
        return all(
            (
                self.supports_metadata_filtering,
                self.supports_namespace_filtering,
                self.supports_time_filtering,
                self.supports_delete_by_filter,
            )
        )


class VectorStore(Protocol):
    @property
    def capabilities(self) -> VectorCapabilities: ...

    def upsert_vector(self, uri: str, embedding: list[float], metadata: dict | None = None) -> None: ...

    def delete_vector(self, uri: str) -> None: ...

    def search_vector(self, embedding: list[float], namespace: str, limit: int = 10) -> list[VectorHit]: ...

    def get_vector_metadata(self, uri: str) -> dict | None: ...

    def vector_uris(self) -> list[str]: ...

    def search_vector_candidates(
        self,
        embedding: list[float],
        candidate_uris: list[str] | tuple[str, ...],
        *,
        limit: int = 10,
    ) -> list[VectorHit]: ...

    def search_vector_filtered(
        self,
        embedding: list[float],
        *,
        namespace: str,
        filters: Mapping[str, object],
        limit: int = 10,
    ) -> list[VectorHit]: ...

    def delete_by_filter(self, filters: Mapping[str, object]) -> int: ...


def vector_row_id(tenant_id: str, catalog_record_key: str) -> str:
    tenant = str(tenant_id or "default")
    record_key = str(catalog_record_key or "")
    if not tenant.strip() or not record_key.strip() or "\x00" in tenant or "\x00" in record_key:
        raise ValueError("vector row identity requires a valid tenant and Catalog record key")
    tenant_digest = hashlib.sha256(tenant.encode("utf-8")).hexdigest()
    record_digest = hashlib.sha256(record_key.encode("utf-8")).hexdigest()
    return f"memoryos-vector://v1/{tenant_digest}/{record_digest}"


def vector_capabilities(store: object) -> VectorCapabilities:
    capabilities = getattr(store, "capabilities", None)
    return capabilities if isinstance(capabilities, VectorCapabilities) else VectorCapabilities()


def require_production_vector_capabilities(store: object) -> VectorCapabilities:
    capabilities = vector_capabilities(store)
    required = {
        "supports_metadata_filtering": capabilities.supports_metadata_filtering,
        "supports_namespace_filtering": capabilities.supports_namespace_filtering,
        "supports_time_filtering": capabilities.supports_time_filtering,
        "supports_delete_by_filter": capabilities.supports_delete_by_filter,
    }
    missing = tuple(name for name, enabled in required.items() if not enabled)
    missing_methods = tuple(
        name for name in ("search_vector_filtered", "delete_by_filter") if not callable(getattr(store, name, None))
    )
    if missing or missing_methods:
        raise ValueError(
            "production VectorStore is missing required native capabilities: "
            + ", ".join((*missing, *missing_methods))
        )
    return capabilities


__all__ = [
    "VectorCapabilities",
    "VectorHit",
    "VectorStore",
    "require_production_vector_capabilities",
    "vector_capabilities",
    "vector_row_id",
]
