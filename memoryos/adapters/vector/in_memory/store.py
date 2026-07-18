"""Bounded in-memory vector adapter for local and test execution."""

from __future__ import annotations

import math
from collections.abc import Mapping

from memoryos.contextdb.store.vector import VectorCapabilities, VectorHit, vector_row_id


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.rows: dict[str, tuple[list[float], dict]] = {}
        self._metadata_rows: dict[tuple[str, object], set[str]] = {}

    @property
    def capabilities(self) -> VectorCapabilities:
        return VectorCapabilities(supports_namespace_filtering=True, supports_delete_by_filter=True)

    def upsert_vector(self, uri: str, embedding: list[float], metadata: dict | None = None) -> None:
        prepared_metadata = dict(metadata or {})
        catalog_record_key = str(prepared_metadata.get("catalog_record_key") or "")
        tenant_id = str(prepared_metadata.get("tenant_id") or "")
        if tenant_id and not catalog_record_key:
            raise ValueError("tenant-scoped vector metadata requires catalog_record_key")
        if catalog_record_key:
            if not tenant_id:
                raise ValueError("Catalog vector metadata requires tenant_id")
            expected_row_id = vector_row_id(tenant_id, catalog_record_key)
            if str(uri) != expected_row_id:
                raise ValueError("Catalog vector row ID must be derived from tenant_id and record_key")
        previous = self.rows.get(uri)
        if previous is not None:
            self._discard_metadata_identity(uri, previous[1])
        self.rows[uri] = (_finite_vector(embedding), prepared_metadata)
        self._index_metadata_identity(uri, prepared_metadata)

    def delete_vector(self, uri: str) -> None:
        removed = self.rows.pop(uri, None)
        if removed is not None:
            self._discard_metadata_identity(uri, removed[1])

    def get_vector_metadata(self, uri: str) -> dict | None:
        row = self.rows.get(uri)
        return dict(row[1]) if row is not None else None

    def vector_uris(self) -> list[str]:
        return list(self.rows)

    def search_vector(self, embedding: list[float], namespace: str, limit: int = 10) -> list[VectorHit]:
        embedding = _finite_vector(embedding)
        hits: list[VectorHit] = []
        for uri, (stored, metadata) in self.rows.items():
            if namespace:
                metadata_namespace = str(metadata.get("namespace") or "")
                tenant_id = str(metadata.get("tenant_id") or "")
                if not (
                    uri.startswith(namespace)
                    or metadata_namespace == namespace
                    or tenant_id == namespace
                ):
                    continue
            hits.append(VectorHit(uri=uri, score=self._cosine(embedding, stored), metadata=metadata))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:limit]

    def search_vector_candidates(
        self,
        embedding: list[float],
        candidate_uris: list[str] | tuple[str, ...],
        *,
        limit: int = 10,
    ) -> list[VectorHit]:
        embedding = _finite_vector(embedding)
        hits: list[VectorHit] = []
        for row_id in tuple(dict.fromkeys(str(item) for item in candidate_uris)):
            row = self.rows.get(row_id)
            if row is not None:
                stored, metadata = row
                hits.append(VectorHit(row_id, self._cosine(embedding, stored), dict(metadata)))
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
        if not str(filters.get("tenant_id") or ""):
            raise ValueError("vector metadata deletion requires tenant_id")
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

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
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


__all__ = ["InMemoryVectorStore"]
