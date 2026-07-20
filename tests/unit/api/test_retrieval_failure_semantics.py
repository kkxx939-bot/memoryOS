from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from infrastructure.context.orchestrator import RetrievalUnavailableError
from infrastructure.context.retrieval.query_plan import RetrievalOptions, RetrievalQueryIntent
from infrastructure.store.contracts.vector import (
    VectorCapabilities,
    VectorHit,
    vector_row_id,
)
from infrastructure.store.model.catalog import CatalogRecord
from infrastructure.store.query import CatalogCandidateBoundExceeded
from openApi.mcp.errors import exception_payload
from openApi.sdk.client import MemoryOSClient
from sanitization.context_projection import (
    ContextProjectionSanitizationError,
    ContextProjectionSanitizer,
)
from tests.support.persistence import InMemoryVectorStore


class _Embedding:
    model_name = "failure-semantics-test"
    dimension = 2

    def embed(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


class _CapturingEmbedding(_Embedding):
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.queries.append(text)
        return super().embed(text)


class _CapturingReranker:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.payloads: list[list[dict[str, Any]]] = []

    def rerank(self, query: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.queries.append(query)
        self.payloads.append(items)
        return items


class _CapturingFilteredVectorStore(InMemoryVectorStore):
    def __init__(self, *, record_key: str, public_uri: str) -> None:
        super().__init__()
        self.record_key = record_key
        self.public_uri = public_uri
        self.filtered_calls = 0

    @property
    def capabilities(self) -> VectorCapabilities:
        return VectorCapabilities(
            supports_metadata_filtering=True,
            supports_namespace_filtering=True,
            supports_time_filtering=True,
            supports_delete_by_filter=True,
        )

    def search_vector_filtered(
        self,
        embedding: list[float],
        *,
        namespace: str,
        filters: Mapping[str, object],
        limit: int = 10,
    ) -> list[VectorHit]:
        del embedding, filters, limit
        self.filtered_calls += 1
        return [
            VectorHit(
                uri=vector_row_id(namespace, self.record_key),
                score=0.95,
                metadata={
                    "catalog_record_key": self.record_key,
                    "tenant_id": namespace,
                    "public_uri": self.public_uri,
                },
            )
        ]


class _BrokenFilteredVectorStore(InMemoryVectorStore):
    @property
    def capabilities(self) -> VectorCapabilities:
        return VectorCapabilities(
            supports_metadata_filtering=True,
            supports_namespace_filtering=True,
            supports_time_filtering=True,
            supports_delete_by_filter=True,
        )

    def search_vector_filtered(
        self,
        embedding: list[float],
        *,
        namespace: str,
        filters: Mapping[str, object],
        limit: int = 10,
    ) -> list[VectorHit]:
        del embedding, namespace, filters, limit
        raise OSError("vector backend unavailable")


class _InvalidFilteredVectorStore(_BrokenFilteredVectorStore):
    def search_vector_filtered(
        self,
        embedding: list[float],
        *,
        namespace: str,
        filters: Mapping[str, object],
        limit: int = 10,
    ) -> Any:
        del embedding, namespace, filters, limit
        return None


def test_vector_and_reranker_provider_egress_sanitizes_query_and_payload(
    tmp_path,
    monkeypatch,
) -> None:  # noqa: ANN001
    record_key = "safe-provider-egress"
    public_uri = "memoryos://user/u1/resources/quarterly-report"
    provider = _CapturingEmbedding()
    reranker = _CapturingReranker()
    vector = _CapturingFilteredVectorStore(record_key=record_key, public_uri=public_uri)
    client = MemoryOSClient(
        str(tmp_path),
        vector_store=vector,
        embedding_provider=provider,
        reranker=reranker,
    )
    timestamp = "2026-07-14T12:00:00+00:00"
    client.runtime.stores.index.upsert_catalog(  # type: ignore[attr-defined]
        CatalogRecord(
            record_key=record_key,
            uri=public_uri,
            tenant_id="default",
            owner_user_id="u1",
            context_type="resource",
            source_kind="resource_reference",
            created_at=timestamp,
            updated_at=timestamp,
            event_time=timestamp,
            ingested_at=timestamp,
            transaction_time=timestamp,
            title="Quarterly report",
            l0_text="quarterly report",
            l1_text="The quarterly report remains semantically searchable.",
        ),
        tenant_id="default",
    )

    def forbidden_online_enumeration(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("online retrieval must not enumerate a whole store")

    monkeypatch.setattr(client.runtime.stores.source, "list_objects", forbidden_online_enumeration)
    monkeypatch.setattr(client.runtime.stores.vector, "vector_uris", forbidden_online_enumeration)
    query = (
        "quarterly report Authorization: Bearer super-secret-token "
        "password=hunter2 /Users/u1/Desktop/quarterly.txt"
    )
    hits = client.search_context(
        query,
        options=RetrievalOptions(query_intent=RetrievalQueryIntent.OPEN_RECALL),
        user_id="u1",
    )

    assert [item["uri"] for item in hits] == [public_uri]
    assert vector.filtered_calls == 1
    assert len(provider.queries) == 1
    assert len(reranker.queries) == 1
    outbound = json.dumps(
        {
            "embedding_queries": provider.queries,
            "reranker_queries": reranker.queries,
            "reranker_payloads": reranker.payloads,
        },
        ensure_ascii=False,
    )
    assert "quarterly report" in outbound
    assert "super-secret-token" not in outbound
    assert "hunter2" not in outbound
    assert "/Users/u1" not in outbound


def test_provider_egress_sanitization_failure_calls_no_provider(
    tmp_path,
    monkeypatch,
) -> None:  # noqa: ANN001
    record_key = "failed-provider-egress"
    public_uri = "memoryos://user/u1/resources/local-fallback"
    provider = _CapturingEmbedding()
    reranker = _CapturingReranker()
    vector = _CapturingFilteredVectorStore(record_key=record_key, public_uri=public_uri)
    client = MemoryOSClient(
        str(tmp_path),
        vector_store=vector,
        embedding_provider=provider,
        reranker=reranker,
    )
    timestamp = "2026-07-14T12:00:00+00:00"
    client.runtime.stores.index.upsert_catalog(  # type: ignore[attr-defined]
        CatalogRecord(
            record_key=record_key,
            uri=public_uri,
            tenant_id="default",
            owner_user_id="u1",
            context_type="resource",
            source_kind="resource_reference",
            created_at=timestamp,
            updated_at=timestamp,
            event_time=timestamp,
            ingested_at=timestamp,
            transaction_time=timestamp,
            title="Local fallback",
            l0_text="local fallback",
            l1_text="local fallback remains available through FTS",
        ),
        tenant_id="default",
    )

    def fail_projection(*_args: Any, **_kwargs: Any) -> Any:
        raise ContextProjectionSanitizationError("forced provider egress failure")

    monkeypatch.setattr(ContextProjectionSanitizer, "sanitize", fail_projection)
    hits = client.search_context(
        "local fallback",
        options=RetrievalOptions(query_intent=RetrievalQueryIntent.OPEN_RECALL),
        user_id="u1",
    )

    assert [item["uri"] for item in hits] == [public_uri]
    assert provider.queries == []
    assert reranker.queries == []
    assert vector.filtered_calls == 0


def test_retrieval_unavailable_is_retryable_at_mcp_boundary() -> None:
    payload = exception_payload(
        RetrievalUnavailableError("projection lag", degraded_modes=("memory_document_projection_pending:1",))
    )

    assert payload["error"]["code"] == "NOT_READY"
    assert payload["error"]["retryable"] is True
    assert payload["error"]["details"]["degraded_modes"] == ["memory_document_projection_pending:1"]


def test_vector_failure_without_bounded_fallback_is_explicitly_unavailable(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(
        str(tmp_path),
        vector_store=_BrokenFilteredVectorStore(),
        embedding_provider=_Embedding(),
    )

    with pytest.raises(RetrievalUnavailableError, match="vector backend failed"):
        client.search_context(
            "vector-only-miss",
            options=RetrievalOptions(query_intent=RetrievalQueryIntent.OPEN_RECALL),
            user_id="u1",
        )


def test_invalid_vector_backend_response_is_not_treated_as_zero_hits(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(
        str(tmp_path),
        vector_store=_InvalidFilteredVectorStore(),
        embedding_provider=_Embedding(),
    )

    with pytest.raises(RetrievalUnavailableError, match="vector backend failed") as caught:
        client.search_context(
            "invalid-vector-response",
            options=RetrievalOptions(query_intent=RetrievalQueryIntent.OPEN_RECALL),
            user_id="u1",
        )

    assert caught.value.degraded_modes == ("vector_fallback:InvalidResponse",)


def test_sqlite_vm_guard_is_not_reported_as_empty_success(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    timestamp = "2026-07-14T00:00:00+00:00"
    records = tuple(
        CatalogRecord(
            record_key=f"vm-bound-{index:04d}",
            uri=f"memoryos://user/u1/resources/vm-bound-{index:04d}",
            tenant_id="default",
            owner_user_id="u1",
            context_type="resource",
            source_kind="resource",
            created_at=timestamp,
            updated_at=timestamp,
            event_time=timestamp,
            ingested_at=timestamp,
            transaction_time=timestamp,
            title=f"Common bounded term {index}",
            l0_text="common bounded term",
            l1_text="common bounded term repeated for the VM guard",
        )
        for index in range(150)
    )
    client.runtime.stores.index.upsert_catalog_batch(records, tenant_id="default")  # type: ignore[attr-defined]
    client.runtime.stores.index.online_vm_step_limit = 1  # type: ignore[attr-defined]

    with pytest.raises(RetrievalUnavailableError, match="exceeded its online bound") as caught:
        client.search_context(
            "common bounded term",
            options=RetrievalOptions(query_intent=RetrievalQueryIntent.OPEN_RECALL),
            user_id="u1",
        )
    assert caught.value.degraded_modes == ("structured_candidate_bound_exhausted",)


def test_required_temporal_structured_fallback_bound_is_explicitly_unavailable(
    tmp_path,
    monkeypatch,
) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    store = client.runtime.stores.index
    monkeypatch.setattr(store, "search_catalog", lambda *_args, **_kwargs: [])

    def exhaust_temporal_fallback(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise CatalogCandidateBoundExceeded("forced temporal structured bound")

    monkeypatch.setattr(store, "list_catalog", exhaust_temporal_fallback)

    with pytest.raises(RetrievalUnavailableError, match="exceeded its online bound") as caught:
        client.search_context(
            "2026年7月14日发生了什么",
            options=RetrievalOptions(timezone="Asia/Singapore"),
            user_id="u1",
        )

    assert caught.value.degraded_modes == ("structured_candidate_bound_exhausted",)


def test_pending_composite_session_job_is_not_catalog_projection_lag(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    result = client.commit_agent_session(
        user_id="u1",
        session_id="projected-but-consumers-pending",
        messages=[{"role": "user", "content": "catalog projection succeeds"}],
        async_commit=False,
    )

    assert result.session_projection_status == "projected"
    assert client.runtime.stores.queue.stats(queue_name="commit").get("pending") == 1
    assembled = client.assemble_context(
        "definitely-not-in-the-projected-session",
        options=RetrievalOptions(query_intent=RetrievalQueryIntent.OPEN_RECALL),
        user_id="u1",
    )
    assert assembled["contexts"]
    assert not any(str(mode).startswith("session_projection_") for mode in assembled["degraded_modes"])
