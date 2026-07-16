"""上下文数据库里的向量存储。"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class VectorHit:
    # ``uri`` is the backend row identity.  Public Context/Source URIs live in
    # metadata and must never be used as a cross-tenant storage key.
    uri: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class VectorCapabilities:
    """Capabilities used by the planner before it chooses a bounded path."""

    supports_metadata_filtering: bool = False
    supports_namespace_filtering: bool = False
    supports_time_filtering: bool = False
    supports_delete_by_filter: bool = False

    @property
    def production_filtered_top_k_ready(self) -> bool:
        """Whether the backend can enforce every serving boundary natively."""

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


class _VectorRows(dict[str, tuple[list[float], dict]]):
    """Storage-key map with read-only unique legacy URI lookup.

    Older local integrations inspected ``rows[public_uri]`` directly.  Keep
    that diagnostic lookup only when it resolves to exactly one row; writes
    and iteration always use the tenant-scoped backend identity, and an
    ambiguous cross-tenant public URI fails closed.
    """

    @staticmethod
    def _public_identities(metadata: Mapping[str, object]) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                str(metadata.get(key) or "")
                for key in ("public_uri", "uri", "source_uri", "claim_uri")
                if metadata.get(key)
            )
        )

    def legacy_key(self, key: object) -> str | None:
        text = str(key)
        matches = [
            row_id for row_id, (_embedding, metadata) in dict.items(self) if text in self._public_identities(metadata)
        ]
        return matches[0] if len(matches) == 1 else None

    def __getitem__(self, key: str) -> tuple[list[float], dict]:
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            resolved = self.legacy_key(key)
            if resolved is None:
                raise
            return dict.__getitem__(self, resolved)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.rows: _VectorRows = _VectorRows()
        self._metadata_rows: dict[tuple[str, object], set[str]] = {}

    @property
    def capabilities(self) -> VectorCapabilities:
        return VectorCapabilities(
            supports_namespace_filtering=True,
            supports_delete_by_filter=True,
        )

    def upsert_vector(self, uri: str, embedding: list[float], metadata: dict | None = None) -> None:
        previous = dict.get(self.rows, uri)
        if previous is not None:
            self._discard_metadata_identity(uri, previous[1])
        prepared_metadata = dict(metadata or {})
        self.rows[uri] = (_finite_vector(embedding), prepared_metadata)
        self._index_metadata_identity(uri, prepared_metadata)

    def delete_vector(self, uri: str) -> None:
        if dict.__contains__(self.rows, uri):
            removed = dict.pop(self.rows, uri, None)
            if removed is not None:
                self._discard_metadata_identity(uri, removed[1])
            return
        legacy_key = self.rows.legacy_key(uri)
        if legacy_key is not None:
            removed = dict.pop(self.rows, legacy_key, None)
            if removed is not None:
                self._discard_metadata_identity(legacy_key, removed[1])

    def get_vector_metadata(self, uri: str) -> dict | None:
        row = dict.get(self.rows, uri)
        if row is None:
            legacy_key = self.rows.legacy_key(uri)
            row = dict.get(self.rows, legacy_key) if legacy_key is not None else None
        return dict(row[1]) if row is not None else None

    def vector_uris(self) -> list[str]:
        return list(self.rows)

    def search_vector(self, embedding: list[float], namespace: str, limit: int = 10) -> list[VectorHit]:
        embedding = _finite_vector(embedding)
        hits = []
        for uri, (stored, metadata) in self.rows.items():
            if namespace:
                public_uri = str(metadata.get("public_uri") or metadata.get("uri") or "")
                metadata_namespace = str(metadata.get("namespace") or "")
                tenant_id = str(metadata.get("tenant_id") or "")
                if not (
                    uri.startswith(namespace)
                    or public_uri.startswith(namespace)
                    or metadata_namespace == namespace
                    or tenant_id == namespace
                ):
                    continue
            score = self._cosine(embedding, stored)
            hits.append(VectorHit(uri=uri, score=score, metadata=metadata))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:limit]

    def search_vector_candidates(
        self,
        embedding: list[float],
        candidate_uris: list[str] | tuple[str, ...],
        *,
        limit: int = 10,
    ) -> list[VectorHit]:
        """Score only the SQL/FTS-bounded candidate set; never enumerate keys."""

        embedding = _finite_vector(embedding)
        hits: list[VectorHit] = []
        for row_id in tuple(dict.fromkeys(str(item) for item in candidate_uris)):
            row = self.rows.get(row_id)
            resolved_row_id = row_id
            if row is None:
                legacy_key = self.rows.legacy_key(row_id)
                if legacy_key is None:
                    continue
                resolved_row_id = legacy_key
                row = self.rows.get(legacy_key)
            if row is None:
                continue
            stored, metadata = row
            hits.append(
                VectorHit(
                    uri=resolved_row_id,
                    score=self._cosine(embedding, stored),
                    metadata=dict(metadata),
                )
            )
        hits.sort(key=lambda item: (-item.score, item.uri))
        return hits[: max(0, int(limit))]

    def search_vector_filtered(
        self,
        embedding: list[float],
        *,
        namespace: str,
        filters: Mapping[str, object],
        limit: int = 10,
    ) -> list[VectorHit]:
        del embedding, namespace, filters, limit
        raise RuntimeError("local VectorStore does not support native filtered Top-K")

    def delete_by_filter(self, filters: Mapping[str, object]) -> int:
        """Delete exact metadata matches through a maintained reverse index.

        This path exists for durable orphan cleanup after the rebuildable
        Catalog row has already disappeared.  It deliberately does not scan
        ``rows`` (and therefore does not turn deletion into a hidden O(N)
        serving fallback).
        """

        normalized = tuple((str(key), self._metadata_index_value(value)) for key, value in filters.items())
        if not normalized:
            raise ValueError("vector metadata deletion requires at least one exact filter")
        matches: set[str] | None = None
        for identity in normalized:
            row_ids = self._metadata_rows.get(identity, set())
            matches = set(row_ids) if matches is None else matches & row_ids
            if not matches:
                return 0
        doomed = tuple(sorted(matches or ()))
        for uri in doomed:
            self.delete_vector(uri)
        return len(doomed)

    @staticmethod
    def _metadata_index_value(value: object) -> object:
        if isinstance(value, str | int | float | bool) or value is None:
            return value
        raise ValueError("vector metadata deletion supports scalar exact filters only")

    def _index_metadata_identity(self, row_id: str, metadata: Mapping[str, object]) -> None:
        for key, value in metadata.items():
            try:
                identity = (str(key), self._metadata_index_value(value))
            except ValueError:
                continue
            self._metadata_rows.setdefault(identity, set()).add(row_id)

    def _discard_metadata_identity(self, row_id: str, metadata: Mapping[str, object]) -> None:
        for key, value in metadata.items():
            try:
                identity = (str(key), self._metadata_index_value(value))
            except ValueError:
                continue
            indexed = self._metadata_rows.get(identity)
            if indexed is None:
                continue
            indexed.discard(row_id)
            if not indexed:
                self._metadata_rows.pop(identity, None)

    def _cosine(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = sum(a * a for a in left) ** 0.5
        right_norm = sum(b * b for b in right) ** 0.5
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _finite_vector(values: list[float]) -> list[float]:
    result = [float(value) for value in values]
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError("vector values must be finite and non-empty")
    return result


def vector_row_id(tenant_id: str, catalog_record_key: str) -> str:
    """Return a tenant- and Catalog-record-scoped rebuildable vector key.

    A public URI is not a storage identity: two tenants may legitimately use
    the same URI and one Catalog object may have several immutable revisions.
    Hashing both trusted identities also avoids leaking tenant or record names
    through backend collection keys.
    """

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
    """Reject a backend that cannot filter before Top-K in production.

    Local stores may deliberately use the bounded SQL/FTS-candidate fallback.
    This check is intentionally explicit so aliases and test doubles are not
    mistaken for a production metadata-filtering vector database.
    """

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
        details = (*missing, *missing_methods)
        raise ValueError("production VectorStore is missing required native capabilities: " + ", ".join(details))
    return capabilities


__all__ = [
    "InMemoryVectorStore",
    "VectorCapabilities",
    "VectorHit",
    "VectorStore",
    "require_production_vector_capabilities",
    "vector_row_id",
    "vector_capabilities",
]
