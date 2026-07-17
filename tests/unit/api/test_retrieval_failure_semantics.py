from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from typing import Any

import pytest

from memoryos.api.mcp.errors import exception_payload
from memoryos.api.sdk.client import MemoryOSClient
from memoryos.contextdb.catalog import CatalogRecord
from memoryos.contextdb.model.context_type import ContextType
from memoryos.contextdb.retrieval.errors import CatalogCandidateBoundExceeded
from memoryos.contextdb.retrieval.orchestrator import RetrievalUnavailableError
from memoryos.contextdb.retrieval.query_plan import (
    CanonicalResolutionMode,
    RetrievalOptions,
    RetrievalQueryIntent,
)
from memoryos.contextdb.session.session_model import SessionArchive
from memoryos.contextdb.store.source_store import QueueJob
from memoryos.contextdb.store.sqlite_index_store import SQLiteIndexStore
from memoryos.contextdb.store.vector_store import (
    InMemoryVectorStore,
    VectorCapabilities,
    VectorHit,
    vector_row_id,
)
from memoryos.security.context_projection import (
    ContextProjectionSanitizationError,
    ContextProjectionSanitizer,
)


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
    client.index_store.upsert_catalog(  # type: ignore[attr-defined]
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
        )
    )

    def forbidden_online_enumeration(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("online retrieval must not enumerate a whole store")

    monkeypatch.setattr(client.source_store, "list_objects", forbidden_online_enumeration)
    monkeypatch.setattr(client.vector_store, "vector_uris", forbidden_online_enumeration)
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
    client.index_store.upsert_catalog(  # type: ignore[attr-defined]
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
        )
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
        RetrievalUnavailableError("projection lag", degraded_modes=("canonical_projection_pending:1",))
    )

    assert payload["error"]["code"] == "NOT_READY"
    assert payload["error"]["retryable"] is True
    assert payload["error"]["details"]["degraded_modes"] == ["canonical_projection_pending:1"]


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


def test_exact_candidate_overflow_is_explicitly_unavailable(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    timestamp = "2026-07-14T00:00:00+00:00"
    records = tuple(
        CatalogRecord(
            record_key=f"exact-overflow-{index:04d}",
            uri=f"memoryos://user/u1/resources/exact-overflow-{index:04d}",
            tenant_id="default",
            owner_user_id="u1",
            context_type="resource",
            source_kind="resource",
            created_at=timestamp,
            updated_at=timestamp,
            event_time=timestamp,
            ingested_at=timestamp,
            transaction_time=timestamp,
            title=f"Exact overflow {index}",
            l0_text="bounded exact identity",
            l1_text="bounded exact identity",
            metadata={"scene_key": "shared-exact-identity"},
        )
        for index in range(901)
    )
    client.index_store.upsert_catalog_batch(records)  # type: ignore[attr-defined]

    with pytest.raises(RetrievalUnavailableError, match="online scan bound") as caught:
        client.search_context(
            "shared-exact-identity",
            options=RetrievalOptions(query_intent=RetrievalQueryIntent.OPEN_RECALL),
            user_id="u1",
        )
    assert caught.value.degraded_modes == ("structured_candidate_bound_exhausted",)


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
    client.index_store.upsert_catalog_batch(records)  # type: ignore[attr-defined]
    client.index_store.online_vm_step_limit = 1  # type: ignore[attr-defined]

    with pytest.raises(RetrievalUnavailableError, match="online scan bound") as caught:
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
    store = client.index_store
    monkeypatch.setattr(store, "search_catalog", lambda *_args, **_kwargs: [])

    def exhaust_temporal_fallback(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise CatalogCandidateBoundExceeded("forced temporal structured bound")

    monkeypatch.setattr(store, "list_catalog", exhaust_temporal_fallback)

    with pytest.raises(RetrievalUnavailableError, match="online scan bound") as caught:
        client.search_context(
            "2026年7月14日发生了什么",
            options=RetrievalOptions(timezone="Asia/Singapore"),
            user_id="u1",
        )

    assert caught.value.degraded_modes == ("structured_candidate_bound_exhausted",)


def test_pending_canonical_projection_makes_current_retrieval_fail_closed(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    client.queue_store.enqueue(
        QueueJob(
            job_id="outbox_pending-current",
            queue_name="memory_projection",
            action="project_memory_committed",
            target_uri="memoryos://user/u1/memories/canonical/slots/pending",
            payload={"transaction_id": "pending-current"},
        )
    )

    with pytest.raises(RetrievalUnavailableError, match="Canonical Current projection"):
        client.search_context("preference", user_id="u1")

    assert (
        client.search_context(
            "ordinary history",
            options=RetrievalOptions(query_intent=RetrievalQueryIntent.OPEN_RECALL),
            user_id="u1",
        )
        == []
    )


def test_explicit_missing_current_slot_is_unavailable_without_changing_ordinary_empty_results(
    tmp_path,
) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    missing_slot_uri = "memoryos://user/u1/memories/canonical/slots/missing-current"

    with pytest.raises(RetrievalUnavailableError, match="Current Slot projection is missing") as caught:
        client.search_context(
            "",
            options=RetrievalOptions(
                target_uris=(missing_slot_uri,),
                context_types=(ContextType.MEMORY,),
                query_intent=RetrievalQueryIntent.CURRENT,
                canonical_resolution_mode=CanonicalResolutionMode.REQUIRE,
            ),
            user_id="u1",
        )

    assert caught.value.degraded_modes == ("missing_canonical_current_projection",)
    assert (
        client.search_context(
            "",
            options=RetrievalOptions(
                target_uris=(missing_slot_uri,),
                context_types=(ContextType.MEMORY,),
                query_intent=RetrievalQueryIntent.CURRENT,
                canonical_resolution_mode=CanonicalResolutionMode.DISABLED,
            ),
            user_id="u1",
        )
        == []
    )
    assert (
        client.search_context(
            "",
            options=RetrievalOptions(
                target_uris=("memoryos://user/u1/resources/missing",),
                query_intent=RetrievalQueryIntent.CURRENT,
            ),
            user_id="u1",
        )
        == []
    )


def test_canonical_projection_health_isolated_by_owner(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    client.queue_store.enqueue(
        QueueJob(
            job_id="outbox-u2-only",
            queue_name="memory_projection",
            action="project_memory_committed",
            target_uri="memoryos://user/u2/memories/canonical/slots/pending",
            payload={"transaction_id": "u2-only"},
        )
    )

    assert client.search_context("u1 current miss", user_id="u1") == []
    with pytest.raises(RetrievalUnavailableError, match="Canonical Current projection") as caught:
        client.search_context("u2 current miss", user_id="u2")
    assert caught.value.degraded_modes == ("canonical_projection_pending:1",)


def test_canonical_projection_health_uses_payload_owner_and_workspace_for_subject_uri(
    tmp_path,
) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    client.queue_store.enqueue(
        QueueJob(
            job_id="outbox-subject-scoped",
            queue_name="memory_projection",
            action="project_memory_committed",
            target_uri="memoryos://user/subject_abc123/memories/canonical/slots/pending",
            payload={
                "transaction_id": "subject-scoped",
                "tenant_id": "default",
                "owner_user_id": "u2",
                "workspace_id": "w2",
            },
        )
    )
    w1 = RetrievalOptions(workspace_ids=("w1",))
    w2 = RetrievalOptions(workspace_ids=("w2",))

    assert client.search_context("u1 w2 current miss", options=w2, user_id="u1") == []
    assert client.search_context("u2 w1 current miss", options=w1, user_id="u2") == []
    with pytest.raises(RetrievalUnavailableError, match="Canonical Current projection") as caught:
        client.search_context("u2 w2 current miss", options=w2, user_id="u2")
    assert caught.value.degraded_modes == ("canonical_projection_pending:1",)


def test_pending_session_projection_cannot_be_hidden_as_empty_success(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    client.index_store.set_session_projection_frontier(  # type: ignore[attr-defined]
        tenant_id="default",
        archive_uri="memoryos://user/u1/sessions/history/pending",
        session_id="pending",
        manifest_digest="",
        status="PENDING",
    )

    with pytest.raises(RetrievalUnavailableError, match="Session projection is lagging") as caught:
        client.search_context(
            "file known only to pending tool result",
            options=RetrievalOptions(query_intent=RetrievalQueryIntent.OPEN_RECALL),
            user_id="u1",
        )

    assert caught.value.degraded_modes == ("session_projection_pending:1",)
    assert (
        client.search_context(
            "canonical-only miss",
            options=RetrievalOptions(context_types=(ContextType.MEMORY,)),
            user_id="u1",
        )
        == []
    )


def test_session_projection_frontier_isolated_by_owner_and_workspace(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    client.index_store.set_session_projection_frontier(  # type: ignore[attr-defined]
        tenant_id="default",
        archive_uri="memoryos://user/u2/sessions/history/u2-w2",
        owner_user_id="u2",
        workspace_id="w2",
        session_id="u2-w2",
        manifest_digest="",
        status="PENDING",
    )
    w1 = RetrievalOptions(
        query_intent=RetrievalQueryIntent.OPEN_RECALL,
        workspace_ids=("w1",),
    )
    w2 = RetrievalOptions(
        query_intent=RetrievalQueryIntent.OPEN_RECALL,
        workspace_ids=("w2",),
    )

    assert client.search_context("u1 w1 miss", options=w1, user_id="u1") == []
    with pytest.raises(RetrievalUnavailableError, match="Session projection is lagging") as caught:
        client.search_context("u2 w2 miss", options=w2, user_id="u2")
    assert caught.value.degraded_modes == ("session_projection_pending:1",)

    client.index_store.set_session_projection_frontier(  # type: ignore[attr-defined]
        tenant_id="default",
        archive_uri="memoryos://user/u2/sessions/history/u2-w2",
        owner_user_id="u2",
        workspace_id="w2",
        session_id="u2-w2",
        manifest_digest="",
        status="PROJECTED",
    )
    client.index_store.set_session_projection_frontier(  # type: ignore[attr-defined]
        tenant_id="default",
        archive_uri="memoryos://user/u1/sessions/history/u1-w2",
        owner_user_id="u1",
        workspace_id="w2",
        session_id="u1-w2",
        manifest_digest="",
        status="PENDING",
    )
    assert client.search_context("u1 w1 miss", options=w1, user_id="u1") == []

    client.index_store.set_session_projection_frontier(  # type: ignore[attr-defined]
        tenant_id="default",
        archive_uri="memoryos://user/u1/sessions/history/u1-unscoped",
        owner_user_id="u1",
        workspace_id="",
        session_id="u1-unscoped",
        manifest_digest="",
        status="PENDING",
    )
    with pytest.raises(RetrievalUnavailableError, match="Session projection is lagging") as caught:
        client.search_context("u1 w1 miss", options=w1, user_id="u1")
    assert caught.value.degraded_modes == ("session_projection_pending:1",)
    assert client.index_store.get_session_projection_frontier_summary(  # type: ignore[attr-defined]
        tenant_id="default",
        owner_user_id="u1",
        workspace_ids=("", "w1"),
    ) == {"PENDING": 1}


def test_session_projection_frontier_schema_upgrade_preserves_owner_health(tmp_path) -> None:  # noqa: ANN001
    path = tmp_path / "frontier-v8.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE session_projection_frontier (
              tenant_id TEXT NOT NULL,
              archive_uri TEXT NOT NULL,
              session_id TEXT NOT NULL,
              manifest_digest TEXT NOT NULL DEFAULT '',
              status TEXT NOT NULL,
              last_error TEXT NOT NULL DEFAULT '',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY(tenant_id, archive_uri)
            )
            """
        )
        conn.execute(
            "INSERT INTO session_projection_frontier VALUES (?, ?, ?, '', 'PENDING', '', ?, ?)",
            (
                "default",
                "memoryos://user/u1/sessions/history/old-pending",
                "old-pending",
                "2026-07-14T00:00:00+00:00",
                "2026-07-14T00:00:00+00:00",
            ),
        )
        conn.execute("PRAGMA user_version = 8")

    store = SQLiteIndexStore(path)
    assert store.catalog_schema_version() == 10
    assert store.get_session_projection_frontier_summary(
        tenant_id="default",
        owner_user_id="u1",
        workspace_ids=("",),
    ) == {"PENDING": 1}
    assert store.get_session_projection_frontier_summary(
        tenant_id="default",
        owner_user_id="u2",
        workspace_ids=("",),
    ) == {}
    with sqlite3.connect(path) as conn:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(session_projection_frontier)")}
        indexes = {str(row[1]) for row in conn.execute("PRAGMA index_list(session_projection_frontier)")}
    assert {"owner_user_id", "workspace_id"} <= columns
    assert "idx_session_projection_frontier_scope_status" in indexes


def test_pending_composite_session_job_is_not_catalog_projection_lag(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    result = client.commit_agent_session(
        user_id="u1",
        session_id="projected-but-consumers-pending",
        messages=[{"role": "user", "content": "catalog projection succeeds"}],
        async_commit=False,
    )

    assert result.session_projection_status == "projected"
    assert client.queue_store.stats(queue_name="session_commit").get("pending") == 1
    assert client.index_store.get_session_projection_frontier_summary(  # type: ignore[attr-defined]
        tenant_id="default"
    ) == {"PROJECTED": 1}
    assembled = client.assemble_context(
        "definitely-not-in-the-projected-session",
        options=RetrievalOptions(query_intent=RetrievalQueryIntent.OPEN_RECALL),
        user_id="u1",
    )
    assert assembled["contexts"]
    assert not any(str(mode).startswith("session_projection_") for mode in assembled["degraded_modes"])


def test_startup_repairs_archive_published_before_projection_job(tmp_path) -> None:  # noqa: ANN001
    first = MemoryOSClient(str(tmp_path))
    archive = SessionArchive(
        user_id="u1",
        session_id="crash-window",
        archive_uri="memoryos://user/u1/sessions/history/crash-window",
        metadata={"tenant_id": "default", "timezone": "UTC"},
        messages=[{"role": "user", "content": "recoverable session evidence"}],
    )
    first.session_archive_store.write_sync_archive(archive)
    first.index_store.set_session_projection_frontier(  # type: ignore[attr-defined]
        tenant_id="default",
        archive_uri=archive.archive_uri,
        session_id=archive.session_id,
        # A hard crash after Evidence publish can leave the pre-write value.
        manifest_digest="",
        status="PENDING",
    )
    assert first.queue_store.stats(queue_name="session_commit") == {}

    restarted = MemoryOSClient(str(tmp_path))

    assert restarted.index_store.get_session_projection_frontier_summary(  # type: ignore[attr-defined]
        tenant_id="default"
    ) == {"PROJECTED": 1}
    assert restarted.queue_store.stats(queue_name="session_commit").get("pending") == 1
    assert restarted.search_context(
        "recoverable session evidence",
        options=RetrievalOptions(query_intent=RetrievalQueryIntent.OPEN_RECALL),
        user_id="u1",
    )


def test_startup_marks_prewrite_frontier_without_evidence_abandoned(tmp_path) -> None:  # noqa: ANN001
    first = MemoryOSClient(str(tmp_path))
    first.index_store.set_session_projection_frontier(  # type: ignore[attr-defined]
        tenant_id="default",
        archive_uri="memoryos://user/u1/sessions/history/not-published",
        session_id="not-published",
        manifest_digest="",
        status="PENDING",
    )

    restarted = MemoryOSClient(str(tmp_path))

    assert restarted.index_store.get_session_projection_frontier_summary(  # type: ignore[attr-defined]
        tenant_id="default"
    ) == {"ABANDONED": 1}


def test_canonical_write_reports_current_slot_projection_failure(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    projector = client.memory_projection_worker.current_slot_projector
    assert projector is not None
    original_project = projector.project

    def fail_current_slot(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise OSError("CurrentSlot backend unavailable")

    monkeypatch.setattr(projector, "project", fail_current_slot)

    remember: dict[str, Any] = {
        "user_id": "u1",
        "memory_type": "preference",
        "content": "I like pistachio ice cream",
        "identity_fields": {"subject": "food", "dimension": "ice_cream_flavor"},
    }
    with pytest.raises(RuntimeError, match="transaction committed.*projection is unavailable"):
        client.remember(**remember)

    # An idempotent semantic retry must drain the same durable outbox instead
    # of falsely reporting success merely because no new canonical operation
    # is required.
    with pytest.raises(RuntimeError, match="transaction committed.*projection is unavailable"):
        client.remember(**remember)

    assert client.queue_store.stats(queue_name="memory_projection").get("pending") == 1
    with pytest.raises(RetrievalUnavailableError, match="Canonical Current projection"):
        client.search_context("pistachio", user_id="u1", context_type="memory")

    monkeypatch.setattr(projector, "project", original_project)
    recovered = client.remember(**remember)
    assert recovered["idempotent_replay"] is True
    assert client.queue_store.stats(queue_name="memory_projection") == {"done": 1}
    assert client.search_context("pistachio", user_id="u1", context_type="memory")


def test_stale_current_slot_candidate_is_explicitly_unavailable(tmp_path) -> None:  # noqa: ANN001
    client = MemoryOSClient(str(tmp_path))
    client.remember(
        user_id="u1",
        memory_type="preference",
        content="I like pistachio ice cream",
        identity_fields={"subject": "food", "dimension": "ice_cream_flavor"},
    )
    path = client.index_store.path  # type: ignore[attr-defined]
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE contexts SET canonical_revision = 999 WHERE record_kind = 'current_slot'"
        )

    with pytest.raises(RetrievalUnavailableError, match="failed bounded authoritative validation") as caught:
        client.search_context("pistachio", user_id="u1", context_type="memory")

    assert caught.value.degraded_modes == ("stale_canonical_current_projection",)
